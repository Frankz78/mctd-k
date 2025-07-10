from typing import Optional, Any, Dict, List, Tuple
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce
import wandb
from PIL import Image
import math

from .df_base import DiffusionForcingBase
from utils.logging_utils import (
    make_trajectory_images,
    get_random_start_goal,
)


class MCTSNode:
    """MCTS node for tree search"""
    def __init__(self, state: torch.Tensor, parent: Optional['MCTSNode'] = None, action: Optional[float] = None):
        # Store state - MCTS nodes don't need gradients
        self.state = state.detach().clone()
        self.parent = parent
        self.action = action  # guidance scale that led to this node
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_child(self, state: torch.Tensor, action: float) -> 'MCTSNode':
        child = MCTSNode(state.detach().clone(), parent=self, action=action)
        self.children.append(child)
        return child


class MCTDPlanning(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        self.env_id = cfg.env_id
        self.action_dim = len(cfg.action_mean)
        self.observation_dim = len(cfg.observation_mean)
        self.use_reward = cfg.use_reward
        self.unstacked_dim = self.observation_dim + self.action_dim + int(self.use_reward)
        cfg.x_shape = (self.unstacked_dim,)
        self.episode_len = cfg.episode_len
        self.n_tokens = self.episode_len // cfg.frame_stack + 1
        self.gamma = cfg.gamma
        self.reward_mean = cfg.reward_mean
        self.reward_std = cfg.reward_std
        self.observation_mean = np.array(cfg.observation_mean[: self.observation_dim])
        self.observation_std = np.array(cfg.observation_std[: self.observation_dim])
        self.action_mean = np.array(cfg.action_mean[: self.action_dim])
        self.action_std = np.array(cfg.action_std[: self.action_dim])
        self.open_loop_horizon = cfg.open_loop_horizon
        self.padding_mode = cfg.padding_mode
        
        # MCTS parameters from configuration file
        self.mcts_simulations = getattr(cfg, 'mcts_simulations', 500)  # Total budget of MCTS steps across entire sequence space
        self.mcts_depth = getattr(cfg, 'mcts_depth', 20)  # Maximum search depth (unified parameter name, replaces original n_subplan)
        self.mcts_c_puct = getattr(cfg, 'mcts_c_puct', 1.414)  # UCB exploration constant
        self.mcts_jumpy_interval = getattr(cfg, 'mcts_jumpy_interval', 10)  # Jumpy denoising interval
        # Note: mcts_subplan_size will be dynamically calculated as total_diffusion_steps / mcts_depth in plan() method
        
        # Guidance scale options (each MCTS edge represents a subplan selection)
        self.guidance_scales = getattr(cfg, 'guidance_scales', [0, 0.1, 0.5, 1, 2])  # List of guidance scales to use
        
        super().__init__(cfg)
        self.plot_end_points = cfg.plot_start_goal and self.guidance_scale != 0

    def _build_model(self):
        mean = list(self.observation_mean) + list(self.action_mean)
        std = list(self.observation_std) + list(self.action_std)
        if self.use_reward:
            mean += [self.reward_mean]
            std += [self.reward_std]
        self.cfg.data_mean = np.array(mean).tolist()
        self.cfg.data_std = np.array(std).tolist()
        
        # Apply MCTD hyperparameters from Table 6
        self.cfg.frame_stack = 10  # The Number of Frame Stack
        self.cfg.causal = False  # Causal Mask: Not Used
        self.cfg.scheduling_matrix = "pyramid"  # Scheduling Matrix: pyramid
        
        # Diffusion parameters from Table 6
        self.cfg.diffusion.stabilization_level = 10  # Stabilization Level
        self.cfg.diffusion.beta_schedule = "linear"  # Beta Schedule: Linear
        self.cfg.diffusion.objective = "pred_x0"  # Diffusion Model Objective: x0-prediction
        self.cfg.diffusion.ddim_sampling_eta = 0.0  # DDIM Sampling eta
        self.cfg.diffusion.sampling_timesteps = 20  # The number of Partial Denoising
        
        # Network architecture parameters from Table 6
        self.cfg.diffusion.architecture.network_size = 128  # Network Size
        self.cfg.diffusion.architecture.num_layers = 12  # The Number of Layers
        self.cfg.diffusion.architecture.attn_heads = 4  # The Number of Attention Heads
        self.cfg.diffusion.architecture.dim_feedforward = 512  # The Feedforward Network Dimension
        
        super()._build_model()

    def _preprocess_batch(self, batch):
        observations, actions, rewards, nonterminals = batch
        batch_size, n_frames = observations.shape[:2]

        observations = observations[..., : self.observation_dim]
        actions = actions[..., : self.action_dim]

        if (n_frames - 1) % self.frame_stack != 0:
            raise ValueError("Number of frames - 1 must be divisible by frame stack size")

        nonterminals = torch.cat([torch.ones_like(nonterminals[:, : self.frame_stack]), nonterminals[:, :-1]], dim=1)
        nonterminals = nonterminals.bool().permute(1, 0)
        masks = torch.cumprod(nonterminals, dim=0).contiguous()

        rewards = rewards[:, :-1, None]
        actions = actions[:, :-1]
        init_obs, observations = torch.split(observations, [1, n_frames - 1], dim=1)
        bundles = self._normalize_x(self.make_bundle(observations, actions, rewards))  # (b t c)
        init_bundle = self._normalize_x(self.make_bundle(init_obs[:, 0]))  # (b c)
        init_bundle[:, self.observation_dim :] = 0  # zero out actions and rewards after normalization
        init_bundle = self.pad_init(init_bundle, batch_first=True)  # (b t c)
        bundles = torch.cat([init_bundle, bundles], dim=1)
        bundles = rearrange(bundles, "b (t fs) ... -> t b fs ...", fs=self.frame_stack)
        bundles = bundles.flatten(2, 3).contiguous()

        if self.cfg.external_cond_dim:
            raise ValueError("external_cond_dim not needed in planning")
        conditions = None

        return bundles, conditions, masks

    def training_step(self, batch, batch_idx):
        xs, conditions, masks = self._preprocess_batch(batch)

        n_tokens, batch_size = xs.shape[:2]

        weights = masks.float()
        if not self.causal:
            # manually mask out entries to train for varying length
            random_terminal = torch.randint(2, n_tokens + 1, (batch_size,), device=self.device)
            random_terminal = nn.functional.one_hot(random_terminal, n_tokens + 1)[:, :n_tokens].bool()
            random_terminal = repeat(random_terminal, "b t -> (t fs) b", fs=self.frame_stack)
            nonterminal_causal = torch.cumprod(~random_terminal, dim=0)
            weights *= torch.clip(nonterminal_causal.float(), min=0.05)
            masks *= nonterminal_causal.bool()

        xs_pred, loss = self.diffusion_model(xs, conditions, noise_levels=self._generate_noise_levels(xs, masks=masks))

        loss = self.reweight_loss(loss, weights)

        if batch_idx % 100 == 0:
            self.log("training/loss", loss, on_step=True, on_epoch=False, sync_dist=True)

        xs = self._unstack_and_unnormalize(xs)[self.frame_stack - 1 :]
        xs_pred = self._unstack_and_unnormalize(xs_pred)[self.frame_stack - 1 :]

        # Visualization, including masked out entries
        if self.global_step % 10000 == 0:
            o, a, r = self.split_bundle(xs_pred)
            trajectory = o.detach().cpu().numpy()[:-1, :8]  # last observation is dummy, sample 8
            images = make_trajectory_images(self.env_id, trajectory, trajectory.shape[1], None, None, False)
            for i, img in enumerate(images):
                self.log_image(
                    f"training_visualization/sample_{i}",
                    Image.fromarray(img),
                )

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation"):
        xs, conditions, _ = self._preprocess_batch(batch)
        _, batch_size, *_ = xs.shape
        if self.guidance_scale == 0:
            namespace += "_no_guidance_random_walk"
        horizon = self.episode_len
        if self.action_dim != 2:
            self.eval_planning(
                batch_size, conditions, horizon, namespace + str(horizon)
            )  # can run planning without environment installation
        self.interact(batch_size, conditions, namespace)  # interact if environment is installation

    def plan(self, start: torch.Tensor, goal: torch.Tensor, horizon: int, conditions: Optional[Any] = None):
        """
        MCTS-based diffusion planning following Algorithm 1 from MCTD paper
        Returns plan history of (m, t, b, c), where the last dim of m is the fully diffused plan
        """
        # Print caller information
        import inspect
        try:
            current_frame = inspect.currentframe()
            if current_frame is not None:
                caller_frame = current_frame.f_back
                if caller_frame is not None:
                    caller_name = caller_frame.f_code.co_name
                    caller_line = caller_frame.f_lineno
                    
                    # Get more context about the caller
                    if caller_name == 'eval_planning':
                        caller_context = "eval_planning (Planning Evaluation)"
                    elif caller_name == 'interact':
                        caller_context = "interact (Environment Interaction)"
                    elif caller_name in ['validation_step', 'test_step']:
                        caller_context = f"{caller_name} (Validation/Test Step)"
                    elif caller_name == 'training_step':
                        caller_context = f"{caller_name} (Training Step)"
                    else:
                        caller_context = f"{caller_name} (Other Call)"
                    
                    print(f"🔍 MCTS Plan Called - Caller: {caller_context}, Line: {caller_line}, Batch Size: {start.shape[0]}, Horizon: {horizon}")
                else:
                    print(f"🔍 MCTS Plan Called - Caller: Unknown (Cannot get call stack), Batch Size: {start.shape[0]}, Horizon: {horizon}")
            else:
                print(f"🔍 MCTS Plan Called - Caller: Unknown (Frame is None), Batch Size: {start.shape[0]}, Horizon: {horizon}")
        except Exception:
            print(f"🔍 MCTS Plan Called - Caller: Unknown (Check failed), Batch Size: {start.shape[0]}, Horizon: {horizon}")
        
        batch_size = start.shape[0]

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        def goal_guidance(x):
            # x is a tensor of shape [t b (fs c)]
            pred = rearrange(x, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
            h_padded = pred.shape[0] - self.frame_stack  # include padding when horizon % frame_stack != 0

            if not self.use_reward:
                # sparse / no reward setting, guide with goal like diffuser
                target = torch.stack([start] * self.frame_stack + [goal] * (h_padded))
                dist = nn.functional.mse_loss(pred, target, reduction="none")  # (t fs) b c

                # guidance weight for observation and action
                weight = np.array(
                    [20] * (self.frame_stack)  # conditoning (aka reconstruction guidance)
                    + [1 for _ in range(horizon)]  # try to reach the goal at any horizon
                    + [0] * (h_padded - horizon)  # don't guide padded entries due to horizon % frame_stack != 0
                )
                weight = torch.from_numpy(weight).float().to(self.device)
                
                dist_o, dist_a, _ = self.split_bundle(dist)  # guidance observation and action with separate weights
                dist_a = torch.sum(dist_a, -1, keepdim=True).sqrt()
                dist_o = reduce(dist_o, "t b (n c) -> t b n", "sum", n=self.observation_dim // 2).sqrt()
                dist_o = torch.tanh(dist_o / 2)  # similar to the "squashed gaussian" in RL, squash to (-1, 1)
                dist = torch.cat([dist_o, dist_a], -1)
                weight = repeat(weight, "t -> t c", c=dist.shape[-1])
                weight[self.frame_stack :, 1:] = 8
                weight[: self.frame_stack, 1:] = 2
                weight = torch.ones_like(dist) * weight[:, None]

                episode_return = -(dist * weight).mean() * 1000
            else:
                # dense reward seeting, guide with reward
                raise NotImplementedError("reward guidance not officially supported yet, although implemented")

            return self.guidance_scale * episode_return

        guidance_fn = goal_guidance if self.guidance_scale else None

        plan_tokens = np.ceil(horizon / self.frame_stack).astype(int)
        pad_tokens = 0 if self.causal else self.n_tokens - plan_tokens - 1
        scheduling_matrix = self._generate_scheduling_matrix(plan_tokens)
        
        if scheduling_matrix is None:
            # Return simple plan if scheduling matrix is not available
            simple_plan = torch.stack([start] * horizon)
            simple_plan = rearrange(simple_plan, "t b c -> 1 t b c")
            return simple_plan
        
        # Initialize plan
        chunk = torch.randn((plan_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        chunk = torch.clamp(chunk, -self.cfg.diffusion.clip_noise, self.cfg.diffusion.clip_noise)
        pad = torch.zeros((pad_tokens, batch_size, *self.x_stacked_shape), device=self.device)
        init_token = rearrange(self.pad_init(start), "fs b c -> 1 b (fs c)")
        plan = torch.cat([init_token, chunk, pad], 0)

        # MCTS Algorithm 1 implementation following "MCTD performs a tree search over the 
        # sequence space with a budget of 500 MCTS steps" (2502.07202v4)
        plan_hist = [plan.detach()[: self.n_tokens - pad_tokens]]
        
        # Initialize MCTS root with initial plan (detached since MCTS doesn't need gradients)
        root = MCTSNode(plan[1 : self.n_tokens - pad_tokens].detach())
        
        total_diffusion_steps = scheduling_matrix.shape[0] - 1
        
        # Dynamically calculate subplan size: evenly distribute total diffusion steps across depth levels
        mcts_subplan_size = max(1, total_diffusion_steps // self.mcts_depth)
        
        # MCTS main loop with total budget of mcts_simulations steps across entire sequence space
        # Budget constraint: Each of the mcts_simulations iterations represents one complete MCTS step
        # (Selection→Expansion→Simulation→Backpropagation) applied to the entire tree structure
        # This is NOT per-timestep but rather the total computational budget for planning
        
        # Add progress bar to display MCTS execution progress and runtime
        tqdm_progress = None
        try:
            from tqdm import tqdm
            tqdm_progress = tqdm(range(self.mcts_simulations), 
                                desc=f"MCTS Tree Search (Budget: {self.mcts_simulations} steps)", 
                                ncols=120,
                                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [ {elapsed}, {remaining},  {rate_fmt}]')
            mcts_iterator = tqdm_progress
        except ImportError:
            # If tqdm is not available, use plain range
            mcts_iterator = range(self.mcts_simulations)
        
        for simulation_idx in mcts_iterator:
            # 1. Selection: Select from root node to leaf node
            leaf_node = self._select(root)
            
            # Check depth limit: if maximum depth is reached, skip expansion
            current_depth = self._get_node_depth(leaf_node)
            if current_depth >= self.mcts_depth:
                # Maximum depth reached, directly simulate at leaf node
                simulation_node = leaf_node
            else:
                # 2. Expansion: Add a child node to leaf node (if not fully expanded)
                expanded_child = None
                if not leaf_node.is_expanded:
                    # Calculate diffusion step corresponding to current depth
                    diffusion_step = min(current_depth * mcts_subplan_size, total_diffusion_steps - 1)
                    expanded_child = self._expand(leaf_node, plan, conditions, scheduling_matrix, 
                                                diffusion_step, pad_tokens, batch_size, guidance_fn)
                
                # 3. Select simulation node
                simulation_node = expanded_child if expanded_child is not None else leaf_node
            
            # 4. Simulation: Evaluate the value of selected node
            value = self._simulate(simulation_node, plan, conditions, scheduling_matrix, 
                                 pad_tokens, batch_size, guidance_fn, mcts_subplan_size, total_diffusion_steps)
            
            # 5. Backpropagation: Backpropagate value to root node
            self._backpropagate(simulation_node, value)
        
        # Close progress bar
        if tqdm_progress is not None:
            tqdm_progress.close()
        
        # Algorithm 1 Line 27: return BESTCHILD(root)
        # Apply the complete optimal path found by MCTS for evaluate and interact
        if root.children:
            # Extract the complete optimal path (sequence of nodes from root to leaf)
            optimal_path = self._extract_optimal_path(root)
            
            # Apply the optimal path: complete diffusion process following the best guidance sequence
            plan = self._apply_optimal_path(plan, optimal_path, conditions, scheduling_matrix, 
                                          pad_tokens, batch_size, guidance_fn, total_diffusion_steps)
            
            # Extract MCTS statistics and best action sequence
            mcts_stats = self._extract_mcts_statistics(root, horizon)
            best_action_sequence = self._extract_best_action_sequence(root, plan_hist[-1], horizon)
            
            # Generate MCTS tree visualization
            tree_visualization = self._create_mcts_tree_visualization(root, mcts_stats)
            
            # Log to wandb
            self._log_mcts_results(mcts_stats, best_action_sequence, tree_visualization)
        else:
            # Fallback: if no children were created during budget-constrained search, apply regular diffusion steps
            for step in range(total_diffusion_steps):
                plan = self._apply_diffusion_step(plan, conditions, scheduling_matrix, step, pad_tokens, batch_size, guidance_fn)
        
        plan_hist.append(plan.detach()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        return plan_hist
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1 following Algorithm 3 from MCTD paper"""
        node = root
        # Continue while node is fully expanded AND not a leaf
        while node.is_expanded and not node.is_leaf():
            total_visits = sum(child.visits for child in node.children) + 1
            node = max(node.children, key=lambda c: self._calculate_ucb_score(c, node, total_visits))
        return node
    
    def _select_meta_action(self, node: MCTSNode) -> Optional[float]:
        """SELECTMETAACTION: Determine guidance level following Algorithm 7"""
        # Find which guidance scales haven't been tried yet
        tried_actions = {child.action for child in node.children}
        available_actions = [gs for gs in self.guidance_scales if gs not in tried_actions]
        
        # If all actions have been tried, mark as fully expanded and return None
        if not available_actions:
            node.is_expanded = True
            return None
        
        # Algorithm 7 Line 2: return UCBSELECTION({NO, LOW, MEDIUM, HIGH})
        # Map guidance_scales to meta-action names: 0=NO, 0.1=LOW, 0.5=MEDIUM, 1=HIGH, 2=VERY_HIGH
        
        # Single action case
        if len(available_actions) == 1:
            return available_actions[0]
        
        # UCB Selection for multiple available actions
        total_visits = sum(child.visits for child in node.children) + 1  # +1 to avoid log(0)
        
        # Find action with highest UCB score
        best_action = max(available_actions, 
                         key=lambda action: self._calculate_ucb_score(action, node, total_visits))
        
        return best_action

    def _create_guidance_function(self, guidance_scale: float, base_guidance_fn):
        """Create guidance function based on scale following Algorithm 7"""
        if guidance_scale == 0 or base_guidance_fn is None:  # gs = NO: Pure exploration or no guidance
            # Return a dummy function that returns zero but maintains gradients
            def dummy_guidance(x):
                # Ensure the returned tensor depends on input x to maintain gradient flow
                return torch.sum(x * 0.0)  # This depends on x but evaluates to 0
            return dummy_guidance
        else:  # gs ≠ NO: Guided sampling with scaled guidance
            def scaled_guidance(x):
                # base_guidance_fn already includes self.guidance_scale, so we need to normalize first
                base_result = base_guidance_fn(x)
                # Remove the original scaling and apply the new one
                normalized_result = base_result / max(self.guidance_scale, 1e-8)
                return guidance_scale * normalized_result
            return scaled_guidance

    def _calculate_ucb_score(self, action_or_child, node: MCTSNode, total_visits: int) -> float:
        """Calculate UCB1 score for either a child node or a potential action"""
        
        if isinstance(action_or_child, MCTSNode):
            # Case 1: Direct child node (for _select method)
            child = action_or_child
            if child.visits == 0:
                return float('inf')
            
            exploitation = child.value / child.visits
            if total_visits <= 1:  # Avoid log(0) or log(1)
                return exploitation
            
            import math
            exploration = self.mcts_c_puct * math.sqrt(math.log(total_visits) / child.visits)
            
            return exploitation + exploration
            
        else:
            # Case 2: Potential action (for _select_meta_action method)
            action = action_or_child
            if action is None:
                return float('-inf')
            
            # Get statistics for this action from existing children
            action_visits = 0
            action_value = 0.0
            
            for child in node.children:
                if child.action == action:
                    action_visits = child.visits
                    action_value = child.value / max(child.visits, 1)
                    break
            
            # If action hasn't been tried, give it infinite score (highest priority)
            if action_visits == 0:
                return float('inf')
            
            # UCB1 formula: exploitation + exploration
            import math
            exploitation = action_value
            exploration = self.mcts_c_puct * math.sqrt(math.log(total_visits) / action_visits)
            
            return exploitation + exploration

    def _denoise_subplan(self, node: MCTSNode, guidance_scale: float, plan: torch.Tensor, 
                        conditions, scheduling_matrix: np.ndarray, step: int, 
                        pad_tokens: int, batch_size: int, guidance_fn) -> torch.Tensor:
        """DENOISESUBPLAN: Generate new subplan using diffusion following Algorithm 7"""
        # Algorithm 7 Line 4: procedure DENOISESUBPLAN(node, gs)
        temp_plan = plan.clone()
        temp_plan[1 : self.n_tokens - pad_tokens] = node.state
        
        # Algorithm 7 Line 5-9: Create appropriate guidance function
        temp_guidance_fn = self._create_guidance_function(guidance_scale, guidance_fn)
        
        # Apply diffusion step with appropriate guidance
        temp_plan = self._apply_diffusion_step(temp_plan, conditions, scheduling_matrix, 
                                             step, pad_tokens, batch_size, temp_guidance_fn)
        
        # Return the new subplan state
        return temp_plan[1 : self.n_tokens - pad_tokens]

    def _expand(self, node: MCTSNode, plan: torch.Tensor, conditions, scheduling_matrix: np.ndarray,
                step: int, pad_tokens: int, batch_size: int, guidance_fn) -> Optional[MCTSNode]:
        """Expansion phase following Algorithm 4 from MCTD paper"""
        # Step 2: gs ← SELECTMETAACTION(node) {Determine guidance level}
        guidance_scale = self._select_meta_action(node)
        
        # If no action available (fully expanded), return None
        if guidance_scale is None:
            return None
        
        # Step 3: child ← DENOISESUBPLAN(node, gs) {Generate new subplan using diffusion}
        child_state = self._denoise_subplan(node, guidance_scale, plan, conditions, 
                                          scheduling_matrix, step, pad_tokens, batch_size, guidance_fn)
        
        # Step 4: ADDCHILD(node, child)
        child = node.add_child(child_state, guidance_scale)
        
        # Mark as fully expanded if all actions have been tried
        if len(node.children) == len(self.guidance_scales):
            node.is_expanded = True
        
        # Step 5: return child
        return child
    
    def _fast_jumpy_denoising(self, node: MCTSNode, plan: torch.Tensor, conditions, 
                             scheduling_matrix: np.ndarray, pad_tokens: int, 
                             batch_size: int, guidance_fn, mcts_subplan_size: int, 
                             total_diffusion_steps: int) -> torch.Tensor:
        """FASTJUMPYDENOISING: Complete denoising from node state to final plan following Algorithm 5"""
        # Start with the node's current state
        temp_plan = plan.clone()
        temp_plan[1 : self.n_tokens - pad_tokens] = node.state
        
        # Calculate the diffusion step corresponding to this node's depth
        current_depth = self._get_node_depth(node)
        start_step = min(current_depth * mcts_subplan_size, total_diffusion_steps - 1)
        
        # Apply remaining diffusion steps from the node's corresponding step onwards
        # This maintains MCTS tree semantics where each node represents a partial denoising state
        for step in range(start_step, total_diffusion_steps):
            temp_plan = self._apply_diffusion_step(temp_plan, conditions, scheduling_matrix, 
                                                 step, pad_tokens, batch_size, guidance_fn)
        
        return temp_plan
    
    def _evaluate_plan(self, full_plan: torch.Tensor, guidance_fn) -> float:
        """
        EVALUATEPLAN: Evaluate the quality of a complete plan following Algorithm 5 and A.5.6 MCTD REWARD FUNCTION
        
        Implements two heuristic rules from A.5.6:
        1. Check for physically impossible large position differences between near states
        2. Give reward when reaching goal with formula: r = (H - t)/H for early reaching
        """
        # Convert full_plan to trajectory format for evaluation
        plan_traj = rearrange(full_plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        plan_traj = plan_traj[self.frame_stack:]  # Remove initial padding
        
        # Extract observations from the plan
        observations, _, _ = self.split_bundle(plan_traj)
        batch_size = observations.shape[1]
        horizon = observations.shape[0]
        
        total_reward = 0.0
        
        # Rule 1: Check for physically impossible large position differences between near states
        position_penalty = 0.0
        if horizon > 1:
            # Extract positions (assuming first 2 dimensions are x, y coordinates)
            positions = observations[:, :, :2]  # Shape: (T, B, 2)
            
            # Calculate position differences between consecutive states
            pos_diffs = torch.diff(positions, dim=0)  # Shape: (T-1, B, 2)
            pos_distances = torch.norm(pos_diffs, dim=2)  # Shape: (T-1, B)
            
            # Define maximum physically reasonable distance per step
            max_step_distance = 0.08  # Adjust based on environment specifics
            
            # Penalty for unrealistic jumps
            large_jumps = pos_distances > max_step_distance
            position_penalty = -large_jumps.float().sum().item() * 2.0  # Heavy penalty
            print(f"Pos distances: {pos_distances}")
            print(f"Position penalty: {position_penalty}")
        
        # Rule 2: Reward for reaching the goal using first_reach metric borrowed from validation_step
        goal_reward = 0.0
        if horizon > 0:
            # Simulate trajectory execution similar to interact() method but without environment
            # Extract trajectory positions for goal distance calculation
            batch_size = observations.shape[1]
            
            # Initialize tracking variables like in interact()
            reached = torch.zeros(batch_size, dtype=torch.bool, device=observations.device)
            first_reach = torch.zeros(batch_size, device=observations.device)
            
            # Goal distance threshold for considering "reached" (reward >= 1.0 in interact())
            goal_distance_threshold = 0.1  # Adjust based on environment
            
            # Simulate step-by-step trajectory execution
            for t in range(horizon):
                current_pos = observations[t, :, :2]  # Current position (x, y)
                
                # Calculate distance to goal (assuming goal is at origin or specific target)
                # This is a simplified version - in practice, you'd need the actual goal position
                # For now, use a heuristic based on guidance function
                if guidance_fn:
                    # Create a plan up to current timestep to evaluate goal distance
                    current_plan = full_plan.clone()
                    # Zero out future timesteps to evaluate current state
                    plan_traj = rearrange(current_plan, "seq b (fs c) -> (seq fs) b c", fs=self.frame_stack)
                    if t + self.frame_stack < plan_traj.shape[0]:
                        plan_traj[t + self.frame_stack + 1:] = 0
                    current_plan = rearrange(plan_traj, "(seq fs) b c -> seq b (fs c)", fs=self.frame_stack)
                    
                    # Use guidance to estimate "reward" at current timestep
                    step_guidance = guidance_fn(current_plan).mean(dim=0)  # Per batch item
                    simulated_reward = torch.tanh(step_guidance / 100.0)  # Normalize to ~[0,1] range
                    
                    # Check if goal is reached (similar to reward >= 1.0 condition)
                    newly_reached = (simulated_reward >= 0.8) & (~reached)
                    reached = reached | newly_reached
                else:
                    # Fallback when no guidance function: assume no goal reached
                    newly_reached = torch.zeros(batch_size, dtype=torch.bool, device=observations.device)
                
                # Update first_reach counter (increment for non-reached samples)
                first_reach += (~reached).float()
            
            # Calculate goal reward using first_reach metric with r = (H - t)/H formula
            if reached.any():
                # For samples that reached the goal, calculate early reaching reward
                avg_first_reach = first_reach[reached].mean().item()
                t_reach = avg_first_reach  # Average first reach time
                goal_reward = (horizon - t_reach) / horizon * 100.0  # r = (H - t)/H
            else:
                # No goal reached, minimal reward
                goal_reward = 0.0
        
        # Combine all reward components
        total_reward = position_penalty + goal_reward
        
        
        return total_reward

    def _simulate(self, node: MCTSNode, plan: torch.Tensor, conditions, 
                 scheduling_matrix: np.ndarray, pad_tokens: int, batch_size: int, 
                 guidance_fn, mcts_subplan_size: int, total_diffusion_steps: int) -> float:
        """Simulation phase following Algorithm 5 from MCTD paper (Jumpy Denoising)"""
        # Step 2: fullPlan ← FASTJUMPYDENOISING(node)
        full_plan = self._fast_jumpy_denoising(node, plan, conditions, scheduling_matrix, 
                                             pad_tokens, batch_size, guidance_fn, mcts_subplan_size, 
                                             total_diffusion_steps)
        
        # Step 3: return EVALUATEPLAN(fullPlan)
        return self._evaluate_plan(full_plan, guidance_fn)
    
    def _backpropagate(self, node: Optional[MCTSNode], reward: float):
        """Backpropagation phase following Algorithm 6 from MCTD paper"""
        # Follow Algorithm 6 exactly
        while node is not None:
            # Step 3: node.visitCount ← node.visitCount + 1
            node.visits += 1
            
            # Step 4: node.value ← node.value + reward
            node.value += reward
            
            # Step 6: node ← node.parent
            node = node.parent

    def _extract_mcts_statistics(self, root: MCTSNode, horizon: int) -> Dict[str, Any]:
        """Extract MCTS tree statistics for analysis"""
        stats = {
            'total_nodes': 0,
            'total_visits': 0,
            'max_depth': 0,
            'children_stats': [],
            'action_distribution': {},
            'value_distribution': []
        }
        
        def traverse_tree(node: MCTSNode, depth: int = 0):
            stats['total_nodes'] += 1
            stats['total_visits'] += node.visits
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            if node.visits > 0:
                stats['value_distribution'].append(node.value / node.visits)
            
            # Record children statistics
            if node.children:
                for child in node.children:
                    child_stat = {
                        'action': child.action,
                        'visits': child.visits,
                        'value': child.value,
                        'avg_value': child.value / max(child.visits, 1),
                        'depth': depth + 1
                    }
                    stats['children_stats'].append(child_stat)
                    
                    # Count action distribution
                    action_key = f"guidance_{child.action}"
                    if action_key not in stats['action_distribution']:
                        stats['action_distribution'][action_key] = 0
                    stats['action_distribution'][action_key] += child.visits
                    
                    # Recursively traverse
                    traverse_tree(child, depth + 1)
        
        traverse_tree(root)
        
        # Calculate additional statistics
        if stats['children_stats']:
            stats['best_action'] = max(stats['children_stats'], key=lambda x: x['visits'])['action']
            stats['avg_visits_per_action'] = {}
            for action, visits in stats['action_distribution'].items():
                children_with_action = [c for c in stats['children_stats'] if c['action'] == action]
                if children_with_action:  # Avoid division by zero
                    stats['avg_visits_per_action'][action] = visits / len(children_with_action)
                else:
                    stats['avg_visits_per_action'][action] = 0
        
        return stats
    
    def _extract_best_action_sequence(self, root: MCTSNode, final_plan: torch.Tensor, horizon: int) -> Dict[str, Any]:
        """Extract the best action sequence from MCTS tree and final plan"""
        # Get the optimal path through the MCTS tree (consistent with _extract_optimal_path)
        optimal_path = self._extract_optimal_path(root)
        
        # Convert optimal path to best_path information
        best_path = []
        for node in optimal_path[1:]:  # Skip root node
            best_path.append({
                'guidance_scale': node.action,
                'visits': node.visits,
                'avg_value': node.value / max(node.visits, 1)
            })
        
        # Extract action sequence from final plan
        # Safely handle tensor shapes, avoid einops errors
        try:
            final_plan_unstacked = self._unstack_and_unnormalize(final_plan.unsqueeze(0))[0]
            final_plan_clipped = final_plan_unstacked[self.frame_stack - 1 : self.frame_stack - 1 + horizon]
            
            if final_plan_clipped.shape[0] > 0:
                observations, actions, rewards = self.split_bundle(final_plan_clipped)
                
                action_sequence = {
                    'actions': actions.detach().cpu().numpy().tolist() if actions is not None else [],
                    'observations': observations.detach().cpu().numpy()[:, :4].tolist() if observations.shape[-1] >= 4 else observations.detach().cpu().numpy().tolist(),
                    'plan_length': final_plan_clipped.shape[0],
                    'mcts_path': best_path,
                    'mcts_depth': len(best_path)
                }
            else:
                action_sequence = {
                    'actions': [],
                    'observations': [],
                    'plan_length': 0,
                    'mcts_path': best_path,
                    'mcts_depth': len(best_path)
                }
        except Exception as e:
            # If tensor processing fails, return basic information
            action_sequence = {
                'actions': [],
                'observations': [],
                'plan_length': 0,
                'mcts_path': best_path,
                'mcts_depth': len(best_path),
                'extraction_error': str(e)
            }
        
        return action_sequence
    
    def _create_mcts_tree_visualization(self, root: MCTSNode, mcts_stats: Dict[str, Any]) -> Optional[Any]:
        """Create MCTS tree visualization using matplotlib and networkx"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import networkx as nx
            from matplotlib.patches import FancyBboxPatch
            import io
            
            # Check if there's enough data for visualization
            if mcts_stats['total_nodes'] <= 1 or not root.children:
                return None  # Skip visualization if only root node or no children
            
            # Create directed graph
            G = nx.DiGraph()
            pos = {}
            node_labels = {}
            node_colors = []
            node_sizes = []
            edge_colors = []
            
            # Find best path for highlighting
            best_path_nodes = set()
            current_node = root
            node_id = 0
            best_path_nodes.add(node_id)
            
            while current_node.children:
                best_child = max(current_node.children, key=lambda c: c.visits)
                for child in current_node.children:
                    if child == best_child:
                        node_id += 1
                        best_path_nodes.add(node_id)
                        break
                current_node = best_child
            
            # Build graph structure
            def add_nodes_recursive(node: MCTSNode, parent_id: Optional[int] = None, depth: int = 0, x_offset: float = 0.0):
                nonlocal node_id
                current_id = node_id if parent_id is None else len(G.nodes)
                
                # Add node
                G.add_node(current_id)
                
                # Position calculation for tree layout
                pos[current_id] = (x_offset, -depth)
                
                # Node label with visits and value info
                if node.visits > 0:
                    avg_value = node.value / node.visits
                    node_labels[current_id] = f"V:{node.visits}\nR:{avg_value:.2f}"
                    if hasattr(node, 'action') and node.action is not None:
                        node_labels[current_id] += f"\nG:{node.action}"
                else:
                    node_labels[current_id] = "V:0\nR:0.0"
                    if hasattr(node, 'action') and node.action is not None:
                        node_labels[current_id] += f"\nG:{node.action}"
                
                # Node color and size based on visits and best path
                if current_id in best_path_nodes:
                    node_colors.append('#FFD700')  # Gold for best path
                elif node.visits > 0:
                    # Color intensity based on visits (more visits = darker blue)
                    max_visits = max([n.visits for n in self._get_all_nodes(root)] + [1])
                    intensity = min(node.visits / max_visits, 1.0)
                    # Generate blue color with varying intensity
                    blue_intensity = int(255 * (0.3 + 0.7 * intensity))
                    node_colors.append(f'#{100:02x}{150:02x}{blue_intensity:02x}')
                else:
                    node_colors.append('#E0E0E0')  # Light gray for unvisited
                
                # Node size based on visits
                base_size = 300
                if node.visits > 0:
                    max_visits = max([n.visits for n in self._get_all_nodes(root)] + [1])
                    size_multiplier = 1 + (node.visits / max_visits) * 2
                    node_sizes.append(base_size * size_multiplier)
                else:
                    node_sizes.append(base_size * 0.5)
                
                # Add edges to children
                if node.children:
                    child_spacing = 2.0 / (len(node.children) + 1)
                    for i, child in enumerate(node.children):
                        child_id = len(G.nodes)
                        child_x = x_offset + (i - len(node.children)/2 + 0.5) * child_spacing
                        
                        # Add edge
                        G.add_edge(current_id, child_id)
                        
                        # Edge color (highlight best path)
                        if current_id in best_path_nodes:
                            best_child = max(node.children, key=lambda c: c.visits)
                            if child == best_child:
                                edge_colors.append('#FFD700')  # Gold for best path
                            else:
                                edge_colors.append('#CCCCCC')  # Gray for other edges
                        else:
                            edge_colors.append('#CCCCCC')
                        
                        # Recursively add child nodes
                        add_nodes_recursive(child, current_id, depth + 1, child_x)
                
                return current_id
            
            # Build the tree
            root_id = add_nodes_recursive(root)
            
            # Create visualization
            plt.figure(figsize=(14, 10))
            plt.clf()
            
            # Draw the graph  
            # Type ignore for networkx drawing functions that accept color lists
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)  # type: ignore
            nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=20,   # type: ignore
                                 arrowstyle='->', alpha=0.6, width=2)
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
            
            # Add title and legend
            plt.title("MCTS Tree Structure\n(V=Visits, R=Avg Reward, G=Guidance Scale)", 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Create legend
            legend_elements = [
                mpatches.Patch(color='#FFD700', label='Best Path'),
                mpatches.Patch(color='#4A90E2', label='Visited Nodes'),
                mpatches.Patch(color='#E0E0E0', label='Unvisited Nodes')
            ]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
            
            # Add statistics text
            stats_text = f"Total Nodes: {mcts_stats['total_nodes']}\n"
            stats_text += f"Total Visits: {mcts_stats['total_visits']}\n"
            stats_text += f"Max Depth: {mcts_stats['max_depth']}\n"
            if 'best_action' in mcts_stats:
                stats_text += f"Best Guidance: {mcts_stats['best_action']}"
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image for wandb
            from PIL import Image
            tree_image = Image.open(buf)
            
            plt.close()  # Clean up
            buf.close()
            
            return tree_image
            
        except Exception as e:
            # Silently handle visualization errors and return None
            return None
    
    def _get_all_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """Helper function to get all nodes in the tree"""
        nodes = [root]
        for child in root.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _get_node_depth(self, node: MCTSNode) -> int:
        """Calculate the depth of a node (root has depth 0)"""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _extract_optimal_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Extract the optimal path from root to leaf node based on visit counts"""
        optimal_path = [root]
        current_node = root
        
        # Follow the path of highest visit counts until reaching a leaf
        while current_node.children:
            # Select child with most visits (most explored = most promising)
            best_child = max(current_node.children, key=lambda c: c.visits)
            optimal_path.append(best_child)
            current_node = best_child
        
        return optimal_path
    
    def _apply_optimal_path(self, plan: torch.Tensor, optimal_path: List[MCTSNode], 
                           conditions, scheduling_matrix: np.ndarray, pad_tokens: int, 
                           batch_size: int, guidance_fn, total_diffusion_steps: int) -> torch.Tensor:
        """Apply the optimal path by executing diffusion steps with the best guidance sequence"""
        # Start with initial plan
        current_plan = plan.clone()
        
        # Calculate steps per depth level
        steps_per_level = max(1, total_diffusion_steps // len(optimal_path)) if len(optimal_path) > 1 else total_diffusion_steps
        
        # Apply diffusion following the optimal guidance sequence
        current_step = 0
        for i, node in enumerate(optimal_path[1:], 1):  # Skip root node
            # Determine guidance scale for this level
            guidance_scale = node.action if hasattr(node, 'action') and node.action is not None else 0.0
            
            # Create guidance function for this level
            level_guidance_fn = self._create_guidance_function(guidance_scale, guidance_fn)
            
            # Apply diffusion steps for this level
            level_steps = min(steps_per_level, total_diffusion_steps - current_step)
            for step_offset in range(level_steps):
                if current_step < total_diffusion_steps:
                    current_plan = self._apply_diffusion_step(current_plan, conditions, scheduling_matrix, 
                                                            current_step, pad_tokens, batch_size, level_guidance_fn)
                    current_step += 1
        
        # Apply any remaining diffusion steps with the final guidance
        if current_step < total_diffusion_steps and optimal_path:
            final_node = optimal_path[-1]
            final_guidance_scale = final_node.action if hasattr(final_node, 'action') and final_node.action is not None else 0.0
            final_guidance_fn = self._create_guidance_function(final_guidance_scale, guidance_fn)
            
            for remaining_step in range(current_step, total_diffusion_steps):
                current_plan = self._apply_diffusion_step(current_plan, conditions, scheduling_matrix, 
                                                        remaining_step, pad_tokens, batch_size, final_guidance_fn)
        
        return current_plan
    
    def _log_mcts_results(self, mcts_stats: Dict[str, Any], action_sequence: Dict[str, Any], tree_visualization: Optional[Any] = None):
        """Log MCTS results to wandb - statistics from budget-constrained tree search"""
        # Log MCTS tree statistics from budget-constrained search
        self.log("mcts/total_nodes", mcts_stats['total_nodes'])
        self.log("mcts/total_visits", mcts_stats['total_visits'])  # Should equal mcts_simulations budget
        self.log("mcts/max_depth", mcts_stats['max_depth'])
        self.log("mcts/avg_visits_per_node", mcts_stats['total_visits'] / max(mcts_stats['total_nodes'], 1))
        
        if 'best_action' in mcts_stats:
            self.log("mcts/best_guidance_scale", mcts_stats['best_action'])
        
        # Log action distribution
        for action, visits in mcts_stats['action_distribution'].items():
            self.log(f"mcts/action_dist/{action}", visits)
        
        # Log value distribution statistics
        if mcts_stats['value_distribution']:
            import numpy as np
            values = np.array(mcts_stats['value_distribution'])
            self.log("mcts/avg_node_value", float(np.mean(values)))
            self.log("mcts/std_node_value", float(np.std(values)))
            self.log("mcts/min_node_value", float(np.min(values)))
            self.log("mcts/max_node_value", float(np.max(values)))
        
        # Log action sequence information
        self.log("mcts/plan_length", action_sequence['plan_length'])
        self.log("mcts/mcts_depth", action_sequence['mcts_depth'])
        self.log("mcts/optimal_path_length", len(action_sequence['mcts_path']))
        
        # Log best path information
        if action_sequence['mcts_path']:
            guidance_sequence = [step['guidance_scale'] for step in action_sequence['mcts_path']]
            # Log guidance sequence statistics
            import numpy as np
            self.log("mcts/optimal_guidance_mean", float(np.mean(guidance_sequence)))
            self.log("mcts/optimal_guidance_std", float(np.std(guidance_sequence)))
            self.log("mcts/optimal_guidance_max", float(np.max(guidance_sequence)))
            
            for i, step in enumerate(action_sequence['mcts_path']):
                self.log(f"mcts/best_path/step_{i}_guidance", step['guidance_scale'])
                self.log(f"mcts/best_path/step_{i}_visits", step['visits'])
                self.log(f"mcts/best_path/step_{i}_value", step['avg_value'])
        
        # Log tree visualization if available
        if tree_visualization is not None:
            self.log_image("mcts/tree_structure", tree_visualization)
        
        # Store detailed results as wandb artifacts if available
        try:
            import wandb
            # Check if we have a valid wandb logger
            if hasattr(self, 'logger') and self.logger is not None:
                logger_experiment = getattr(self.logger, 'experiment', None)
                if logger_experiment is not None and hasattr(logger_experiment, 'log'):
                    # Create a table for children statistics
                    if mcts_stats['children_stats']:
                        children_table = wandb.Table(
                            columns=['action', 'visits', 'value', 'avg_value', 'depth'],
                            data=[[c['action'], c['visits'], c['value'], c['avg_value'], c['depth']] 
                                  for c in mcts_stats['children_stats']]
                        )
                        logger_experiment.log({"mcts/children_statistics": children_table})
                    
                    # Log action sequence as artifact
                    if action_sequence['actions']:
                        action_artifact = wandb.Artifact("best_action_sequence", type="plan")
                        with action_artifact.new_file("action_sequence.json", mode="w") as f:
                            import json
                            json.dump(action_sequence, f, indent=2)
                        logger_experiment.log_artifact(action_artifact)
                        
        except Exception:
            # Silently handle wandb logging errors
            pass

    def _apply_diffusion_step(self, plan: torch.Tensor, conditions, scheduling_matrix: np.ndarray,
                             step: int, pad_tokens: int, batch_size: int, guidance_fn) -> torch.Tensor:
        """Apply a single diffusion denoising step"""
        stabilization = 0
        from_noise_levels = np.concatenate([
            np.array((stabilization,), dtype=np.int64),
            scheduling_matrix[step],
            np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
        ])
        to_noise_levels = np.concatenate([
            np.array((stabilization,), dtype=np.int64),
            scheduling_matrix[step + 1],
            np.array([self.sampling_timesteps] * pad_tokens, dtype=np.int64),
        ])
        from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
        to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)
        from_noise_levels = repeat(from_noise_levels, "t -> t b", b=batch_size)
        to_noise_levels = repeat(to_noise_levels, "t -> t b", b=batch_size)
        
        # Note: gradients are handled automatically in diffusion model when guidance_fn is used
        
        plan[1 : self.n_tokens - pad_tokens] = self.diffusion_model.sample_step(
            plan, conditions, from_noise_levels, to_noise_levels, guidance_fn=guidance_fn
        )[1 : self.n_tokens - pad_tokens]
        
        return plan

    def eval_planning(self, batch_size: int, conditions=None, horizon=None, namespace="validation"):
        start, goal = get_random_start_goal(self.env_id, batch_size)

        start_normalized = torch.from_numpy(start).float().to(self.device)
        start_normalized = torch.cat([start_normalized, torch.zeros_like(start_normalized)], -1)
        start_normalized = start_normalized[:, : self.observation_dim]
        start_normalized = self.split_bundle(self._normalize_x(self.make_bundle(start_normalized)))[0]

        goal_normalized = torch.from_numpy(goal).float().to(self.device)
        goal_normalized = torch.cat([goal_normalized, torch.zeros_like(goal_normalized)], -1)
        goal_normalized = goal_normalized[:, : self.observation_dim]
        goal_normalized = self.split_bundle(self._normalize_x(self.make_bundle(goal_normalized)))[0]

        horizon = self.episode_len if horizon is None else horizon
        plan_hist = self.plan(start_normalized, goal_normalized, horizon, conditions)
        plan = self._unnormalize_x(plan_hist[-1])
        plan = plan[self.frame_stack - 1 :]

        # Visualization
        o, _, _ = self.split_bundle(plan)
        o = o.detach().cpu().numpy()[:-1, :16]  # last observation is dummy
        images = make_trajectory_images(self.env_id, o, o.shape[1], start, goal, self.plot_end_points)
        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_plan/sample_{i}",
                Image.fromarray(img),
            )

    def interact(self, batch_size: int, conditions=None, namespace="validation"):
        try:
            import d4rl
            import gym
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("d4rl import not successful, skipping environment interaction. Check d4rl installation.")
            return

        print("Interacting with environment... This may take a couple minutes.")

        use_diffused_action = False
        if self.action_dim != 2:
            # https://arxiv.org/abs/2205.09991
            print("Detected reduced observation/action space, using Diffuser like controller.")
        else:
            print("Detected full observation/action space, using MPC controller w/ diffused actions.")
            use_diffused_action = True

        envs = DummyVecEnv([lambda: gym.make(self.env_id)] * batch_size)
        envs.seed(0)

        terminate = False
        obs_mean = self.data_mean[: self.observation_dim]
        obs_std = self.data_std[: self.observation_dim]
        obs = envs.reset()

        obs = torch.from_numpy(obs).float().to(self.device)
        start = obs.detach()
        obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

        goal = np.concatenate(envs.get_attr("goal_locations"))
        goal = torch.Tensor(goal).float().to(self.device)
        goal = torch.cat([goal, torch.zeros_like(goal)], -1)
        goal = goal[:, : self.observation_dim]
        goal_normalized = ((goal - obs_mean[None]) / obs_std[None]).detach()

        steps = 0
        episode_reward = np.zeros(batch_size)
        episode_reward_if_stay = np.zeros(batch_size)
        reached = np.zeros(batch_size, dtype=bool)
        first_reach = np.zeros(batch_size)

        trajectory = []  # actual trajectory
        all_plan_hist = []  # a list of plan histories, each history is a collection of m diffusion steps

        # run mpc with diffused actions
        while not terminate and steps < self.episode_len:
            plan_hist = self.plan(obs_normalized, goal_normalized, self.episode_len - steps, conditions)
            plan_hist = self._unnormalize_x(plan_hist)  # (m t b c)
            plan = plan_hist[-1]  # (t b c)

            all_plan_hist.append(plan_hist.cpu())

            for t in range(self.open_loop_horizon):
                if use_diffused_action:
                    _, action, _ = self.split_bundle(plan[t])
                else:
                    # Convert obs to tensor if needed
                    if not isinstance(obs, torch.Tensor):
                        obs_tensor = torch.from_numpy(obs).float().to(self.device)
                    else:
                        obs_tensor = obs
                    
                    # Extract position and velocity from plan and observation
                    plan_obs, _, _ = self.split_bundle(plan[t])
                    plan_pos = plan_obs[:, :2]
                    obs_pos = obs_tensor[:, :2]
                    
                    if t > 0:
                        plan_obs_prev, _, _ = self.split_bundle(plan[t - 1])
                        plan_vel = plan_pos - plan_obs_prev[:, :2]
                    else:
                        plan_vel = plan_pos - obs_pos
                    
                    obs_vel = obs_tensor[:, 2:4] if obs_tensor.shape[1] > 2 else torch.zeros_like(obs_pos)
                    action = 12.5 * (plan_pos - obs_pos) + 1.2 * (plan_vel - obs_vel)
                action = torch.clip(action, -1, 1).detach().cpu()
                obs, reward, done, _ = envs.step(np.nan_to_num(action.numpy()))

                reached = np.logical_or(reached, reward >= 1.0)
                episode_reward += reward
                episode_reward_if_stay += np.where(~reached, reward, 1)
                first_reach += ~reached

                if done.any():
                    terminate = True
                    break

                obs, reward, done = [torch.from_numpy(item).float() for item in [obs, reward, done]]
                bundle = self.make_bundle(obs, action, reward[..., None])
                trajectory.append(bundle)
                obs = obs.to(self.device)
                obs_normalized = ((obs[:, : self.observation_dim] - obs_mean[None]) / obs_std[None]).detach()

                steps += 1

        self.log(f"{namespace}/episode_reward", episode_reward.mean())
        self.log(f"{namespace}/episode_reward_if_stay", episode_reward_if_stay.mean())
        self.log(f"{namespace}/first_reach", first_reach.mean())

        # Visualization
        samples = min(16, batch_size)
        trajectory = torch.stack(trajectory)
        start = start[:, :2].cpu().numpy().tolist()
        goal = goal[:, :2].cpu().numpy().tolist()
        images = make_trajectory_images(self.env_id, trajectory, samples, start, goal, self.plot_end_points)

        for i, img in enumerate(images):
            self.log_image(
                f"{namespace}_interaction/sample_{i}",
                Image.fromarray(img),
            )

    def pad_init(self, x, batch_first=False):
        x = repeat(x, "b ... -> fs b ...", fs=self.frame_stack).clone()
        if self.padding_mode == "zero":
            x[: self.frame_stack - 1] = 0
        elif self.padding_mode != "same":
            raise ValueError("init_pad must be 'zero' or 'same'")
        if batch_first:
            x = rearrange(x, "fs b ... -> b fs ...")

        return x

    def split_bundle(self, bundle):
        if self.use_reward:
            return torch.split(bundle, [self.observation_dim, self.action_dim, 1], -1)
        else:
            o, a = torch.split(bundle, [self.observation_dim, self.action_dim], -1)
            return o, a, None

    def make_bundle(
        self,
        obs: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ):
        valid_value = None
        if obs is not None:
            valid_value = obs
        if action is not None and valid_value is not None:
            valid_value = action
        if reward is not None and valid_value is not None:
            valid_value = reward
        if valid_value is None:
            raise ValueError("At least one of obs, action, reward must be provided")
        batch_shape = valid_value.shape[:-1]

        if obs is None:
            obs = torch.zeros(batch_shape + (self.observation_dim,)).to(valid_value)
        if action is None:
            action = torch.zeros(batch_shape + (self.action_dim,)).to(valid_value)
        if reward is None:
            reward = torch.zeros(batch_shape + (1,)).to(valid_value)

        bundle = [obs, action]
        if self.use_reward:
            bundle += [reward]

        return torch.cat(bundle, -1)

    def _generate_noise_levels(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise_levels = super()._generate_noise_levels(xs, masks)
        _, batch_size, *_ = xs.shape

        # first frame is almost always known, this reflect that
        if random() < 0.5:
            noise_levels[0] = torch.randint(0, self.timesteps // 4, (batch_size,), device=xs.device)

        return noise_levels
