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
        self.state = state  # current plan state
        self.parent = parent
        self.action = action  # guidance scale that led to this node
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def add_child(self, state: torch.Tensor, action: float) -> 'MCTSNode':
        child = MCTSNode(state, parent=self, action=action)
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
        
        # MCTS parameters (simplified based on Table 6)
        self.n_subplan = getattr(cfg, 'n_subplan', 5)  # Number of subplans for MCTS
        self.guidance_scales = getattr(cfg, 'guidance_scales', [0, 0.1, 0.5, 1, 2])  # List of guidance scales to use
        
        # MCTS algorithm constants - centralized configuration
        self.ucb_c_puct = getattr(cfg, 'ucb_c_puct', 1.41)  # UCB exploration constant
        
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

        # MCTS Algorithm 1 implementation
        plan_hist = [plan.detach()[: self.n_tokens - pad_tokens]]
        
        # Initialize MCTS root with initial plan
        root = MCTSNode(plan[1 : self.n_tokens - pad_tokens].clone())
        
        total_diffusion_steps = scheduling_matrix.shape[0] - 1
        subplan_size = max(1, total_diffusion_steps // self.n_subplan)
        
        # Main MCTS loop for each subplan
        for subplan_idx in range(self.n_subplan):
            step_start = subplan_idx * subplan_size
            step_end = min((subplan_idx + 1) * subplan_size, total_diffusion_steps)
            
            # Skip if no steps in this subplan
            if step_start >= total_diffusion_steps:
                break
            
            # Standard MCTS procedure (Algorithm 1 lines 3-25)
            # 1. Selection: traverse tree to find leaf node
            leaf_node = self._select(root)
            
            # 2. Expansion: add one child if leaf is not fully expanded
            expanded_child = None
            if not leaf_node.is_expanded:
                # Use the middle step of the subplan for expansion
                step = (step_start + step_end) // 2
                expanded_child = self._expand(leaf_node, plan, conditions, scheduling_matrix, step, pad_tokens, batch_size, guidance_fn)
            
            # 3. Simulation: evaluate the expanded node (or leaf if no expansion)
            simulation_node = expanded_child if expanded_child is not None else leaf_node
            
            value = self._simulate(simulation_node, plan, conditions, scheduling_matrix, pad_tokens, batch_size, guidance_fn)
            
            # 4. Backpropagation: update values up the tree
            self._backpropagate(simulation_node, value)
        
        # Algorithm 1 Line 27: return BESTCHILD(root)
        # Select the best child based on visit counts (most explored)
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            # Apply the chosen action (update plan with best child's state)
            plan[1 : self.n_tokens - pad_tokens] = best_child.state
        else:
            # Fallback: if no children were created, apply regular diffusion steps
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
        if guidance_scale == 0:  # gs = NO: Pure exploration
            return lambda x: torch.zeros(1, device=x.device)
        elif base_guidance_fn is None:  # No base guidance available
            return lambda x: torch.zeros(1, device=x.device)
        else:  # gs ≠ NO: Guided sampling with scaled guidance
            return lambda x: guidance_scale * base_guidance_fn(x) / max(self.guidance_scale, 1e-8)

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
            exploration = self.ucb_c_puct * math.sqrt(math.log(total_visits) / child.visits)
            
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
            exploration = self.ucb_c_puct * math.sqrt(math.log(total_visits) / action_visits)
            
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
                             batch_size: int, guidance_fn) -> torch.Tensor:
        """FASTJUMPYDENOISING: Complete denoising from node state to final plan following Algorithm 5"""
        # Start with the node's current state
        temp_plan = plan.clone()
        temp_plan[1 : self.n_tokens - pad_tokens] = node.state
        
        # Apply remaining diffusion steps to complete denoising
        total_steps = scheduling_matrix.shape[0] - 1
        for step in range(total_steps):
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
            max_step_distance = 0.5  # Adjust based on environment specifics
            
            # Penalty for unrealistic jumps
            large_jumps = pos_distances > max_step_distance
            position_penalty = -large_jumps.float().sum().item() * 10.0  # Heavy penalty
        
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
                 guidance_fn) -> float:
        """Simulation phase following Algorithm 5 from MCTD paper (Jumpy Denoising)"""
        # Step 2: fullPlan ← FASTJUMPYDENOISING(node)
        full_plan = self._fast_jumpy_denoising(node, plan, conditions, scheduling_matrix, 
                                             pad_tokens, batch_size, guidance_fn)
        
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
