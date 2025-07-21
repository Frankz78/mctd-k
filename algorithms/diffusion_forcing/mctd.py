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
        self.mcts_annealing_alpha = getattr(cfg, 'mcts_annealing_alpha', 0.5)  # Parameter annealing exponent
        # Note: mcts_subplan_size will be dynamically calculated as total_diffusion_steps / mcts_depth in plan() method
        
        # Manual optimal path parameters
        self.use_manual_path = getattr(cfg, 'use_manual_path', False)  # Whether to use manual path
        self.manual_optimal_path = getattr(cfg, 'manual_optimal_path', None)  # Manual guidance scale sequence
        
        # Guidance scale options (each MCTS edge represents a subplan selection)
        self.guidance_scales = getattr(cfg, 'guidance_scales', [0, 0.1, 0.5, 1, 2])  # List of guidance scales to use
        
        # Debug and logging switches
        # Usage examples in config:
        # enable_tqdm: false                    # Disable progress bar
        # enable_debug_output: false            # Disable debug print statements
        self.enable_tqdm = getattr(cfg, 'enable_tqdm', True)  # Enable/disable tqdm progress bar
        self.enable_debug_output = getattr(cfg, 'enable_debug_output', True)  # Enable/disable debug print statements
        
        # MCTS context flag for proper guidance function handling
        self._in_mcts_context = False
        
        # Guidance function cache for consistency across MCTS search and final execution
        self._guidance_cache = {}
        
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
        self.cfg.diffusion.sampling_timesteps = 50  # The number of Partial Denoising
        
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
        
        Modified to perform independent MCTS searches for each batch element
        """
        # Print caller information
        import inspect
        if self.enable_debug_output:
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
        
        # Clear guidance cache at the start of each plan call
        self._guidance_cache.clear()
        
        # Set MCTS context flag for proper guidance function creation
        self._in_mcts_context = True

        start = self.make_bundle(start)
        goal = self.make_bundle(goal)

        def goal_guidance(x):
            """Goal guidance function that applies global guidance_scale for non-MCTS usage"""
            base_guidance = self._base_goal_guidance(x, start, goal, horizon)
            return self.guidance_scale * base_guidance

        # For MCTS: always create base guidance function, scale will be applied by _create_guidance_function
        # For non-MCTS: use scaled guidance function based on global guidance_scale
        if hasattr(self, '_in_mcts_context') and self._in_mcts_context:
            # MCTS context: provide unscaled base guidance function with proper closure
            def create_base_guidance(s, g, h):
                def base_guidance_fn(x):
                    return self._base_goal_guidance(x, s, g, h)
                return base_guidance_fn
            base_guidance_fn = create_base_guidance(start, goal, horizon)
            guidance_fn = base_guidance_fn
        else:
            # Non-MCTS context: use global guidance_scale
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

        # MCTS Algorithm 1 implementation with independent searches for each batch element
        plan_hist = [plan.detach()[: self.n_tokens - pad_tokens]]
        
        # Initialize separate MCTS roots for each batch element
        roots = []
        for batch_idx in range(batch_size):
            # Extract plan state for this batch element
            batch_plan_state = plan[1 : self.n_tokens - pad_tokens, batch_idx:batch_idx+1].detach()
            root = MCTSNode(batch_plan_state)
            roots.append(root)
        
        total_diffusion_steps = scheduling_matrix.shape[0] - 1
        
        # Dynamically calculate subplan size: evenly distribute total diffusion steps across depth levels
        mcts_subplan_size = max(1, total_diffusion_steps // self.mcts_depth)
        
        # Check if we should use manual optimal path instead of MCTS search
        if self.use_manual_path and self.manual_optimal_path is not None:
            
            # Create manual optimal path for each batch element
            manual_optimal_paths = []
            for batch_idx in range(batch_size):
                # Use the first batch element's state as dummy_state
                dummy_state = plan[1 : self.n_tokens - pad_tokens, batch_idx:batch_idx+1].detach()
                manual_path = self._create_manual_optimal_path(self.manual_optimal_path, dummy_state)
                manual_optimal_paths.append(manual_path)
            
            # Apply manual optimal paths directly
            final_plan = plan.clone()
            for batch_idx in range(batch_size):
                batch_start = start[batch_idx:batch_idx+1]
                batch_goal = goal[batch_idx:batch_idx+1]
                def create_manual_batch_guidance_fn(b_start, b_goal, h):
                    def batch_guidance_fn(x):
                        return self._base_goal_guidance(x, b_start, b_goal, h)
                    return batch_guidance_fn
                batch_guidance_fn = create_manual_batch_guidance_fn(batch_start, batch_goal, horizon)
                
                batch_plan = plan[:, batch_idx:batch_idx+1]
                optimized_batch_plan = self._apply_optimal_path_batch(batch_plan, manual_optimal_paths[batch_idx], 
                                                                    conditions, scheduling_matrix, 
                                                                    pad_tokens, batch_guidance_fn, total_diffusion_steps, batch_idx)
                
                final_plan[:, batch_idx:batch_idx+1] = optimized_batch_plan
            
            # Clear MCTS context flag
            self._in_mcts_context = False
            
            # Return result with manual path applied
            plan_hist.append(final_plan.detach()[: self.n_tokens - pad_tokens])
            plan_hist = torch.stack(plan_hist, 0)
            plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
            return plan_hist
        
        # Add progress bar to display MCTS execution progress and runtime
        tqdm_progress = None
        if self.enable_tqdm:
            try:
                from tqdm import tqdm
                total_steps = self.mcts_simulations * batch_size
                # Try to detect terminal width for better formatting
                import shutil
                try:
                    terminal_width = shutil.get_terminal_size().columns
                    ncols = min(140, max(80, terminal_width - 20))  # Ensure reasonable width
                except:
                    ncols = 120  # Fallback width
                
                tqdm_progress = tqdm(total=total_steps, 
                                    desc=f"MCTS Search (B={batch_size})", 
                                    ncols=ncols,
                                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                                    disable=False)
            except (ImportError, Exception) as e:
                # If tqdm is not available or fails to initialize, disable it
                if self.enable_debug_output:
                    print(f"⚠️ TQDM initialization failed: {e}, disabling progress bar")
                tqdm_progress = None
        
        # Run independent MCTS searches for each batch element
        simulation_count = 0
        for simulation_idx in range(self.mcts_simulations):
            # Calculate batch-level statistics for progress display
            batch_visits = []
            batch_rewards = []
            
            for batch_idx in range(batch_size):
                simulation_count += 1
                
                # Update progress bar with batch statistics (only update every few iterations to avoid spam)
                if tqdm_progress is not None and (simulation_count % max(1, batch_size // 4) == 0 or simulation_count == 1):
                    # Collect visits and rewards from all roots
                    current_visits = []
                    current_rewards = []
                    
                    for root_idx, root in enumerate(roots):
                        if root.visits > 0:
                            current_visits.append(root.visits)
                            avg_reward = root.value / root.visits if root.visits > 0 else 0.0
                            current_rewards.append(avg_reward)
                    
                    # Calculate statistics
                    if current_visits:
                        visits_min, visits_max = min(current_visits), max(current_visits)
                        reward_min, reward_max = min(current_rewards), max(current_rewards)
                        batch_visits.extend(current_visits)
                        batch_rewards.extend(current_rewards)
                    else:
                        visits_min = visits_max = 0
                        reward_min = reward_max = 0.0
                    
                    # Update progress bar postfix
                    progress_info = f"Sim:{simulation_idx+1}/{self.mcts_simulations}, V:[{visits_min}-{visits_max}], R:[{reward_min:.2f}-{reward_max:.2f}]"
                    try:
                        tqdm_progress.set_postfix_str(progress_info)
                    except Exception as e:
                        if self.enable_debug_output:
                            print(f"⚠️ TQDM postfix update failed: {e}")
                        # Don't disable tqdm for postfix failures, just continue
                
                # Always update progress bar counter
                if tqdm_progress is not None:
                    try:
                        tqdm_progress.update(1)
                    except Exception as e:
                        if self.enable_debug_output:
                            print(f"⚠️ TQDM update failed: {e}")
                        # Disable tqdm if it fails
                        tqdm_progress = None
                
                root = roots[batch_idx]
                
                # Create batch-specific guidance function with proper closure
                batch_start = start[batch_idx:batch_idx+1]
                batch_goal = goal[batch_idx:batch_idx+1]
                def create_batch_guidance_fn(b_start, b_goal, h):
                    def batch_guidance_fn(x):
                        return self._base_goal_guidance(x, b_start, b_goal, h)
                    return batch_guidance_fn
                batch_guidance_fn = create_batch_guidance_fn(batch_start, batch_goal, horizon)
                
                # Create batch-specific plan
                batch_plan = plan[:, batch_idx:batch_idx+1]
                
                # 1. Selection: Select from root node to leaf node
                current_sim = simulation_idx * batch_size + batch_idx
                leaf_node = self._select(root, current_sim, self.mcts_simulations * batch_size)
                
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
                        expanded_child = self._expand_batch(leaf_node, batch_plan, conditions, scheduling_matrix, 
                                                    diffusion_step, pad_tokens, batch_guidance_fn, mcts_subplan_size,
                                                    total_diffusion_steps, current_sim, self.mcts_simulations * batch_size)
                    
                    # 3. Select simulation node
                    simulation_node = expanded_child if expanded_child is not None else leaf_node
                
                # 4. Simulation: Evaluate the value of selected node
                value = self._simulate_batch(simulation_node, batch_plan, conditions, scheduling_matrix, 
                                     pad_tokens, batch_guidance_fn, mcts_subplan_size, total_diffusion_steps,
                                     batch_start, batch_goal)
                
                # 5. Backpropagation: Backpropagate value to root node
                self._backpropagate(simulation_node, value)
        
        # Close progress bar
        if tqdm_progress is not None:
            try:
                tqdm_progress.close()
            except Exception as e:
                if self.enable_debug_output:
                    print(f"⚠️ TQDM close failed: {e}")
        
        # Apply optimal paths for each batch element independently
        final_plan = plan.clone()
        for batch_idx in range(batch_size):
            root = roots[batch_idx]
            
            # Create batch-specific guidance function with proper closure
            batch_start = start[batch_idx:batch_idx+1]
            batch_goal = goal[batch_idx:batch_idx+1]
            def create_final_batch_guidance_fn(b_start, b_goal, h):
                def batch_guidance_fn(x):
                    return self._base_goal_guidance(x, b_start, b_goal, h)
                return batch_guidance_fn
            batch_guidance_fn = create_final_batch_guidance_fn(batch_start, batch_goal, horizon)
            
            if root.children:
                # Extract the complete optimal path (sequence of nodes from root to leaf)
                optimal_path = self._extract_optimal_path(root)
                
                # Apply the optimal path: complete diffusion process following the best guidance sequence
                batch_plan = plan[:, batch_idx:batch_idx+1]
                optimized_batch_plan = self._apply_optimal_path_batch(batch_plan, optimal_path, conditions, scheduling_matrix, 
                                              pad_tokens, batch_guidance_fn, total_diffusion_steps, batch_idx)
                
                # Update the final plan with the optimized batch plan
                final_plan[:, batch_idx:batch_idx+1] = optimized_batch_plan
            else:
                # Fallback: if no children were created during budget-constrained search, apply regular diffusion steps
                batch_plan = plan[:, batch_idx:batch_idx+1]
                def create_fallback_batch_guidance_fn(b_start, b_goal, h):
                    def batch_guidance_fn(x):
                        return self._base_goal_guidance(x, b_start, b_goal, h)
                    return batch_guidance_fn
                batch_guidance_fn = create_fallback_batch_guidance_fn(batch_start, batch_goal, horizon)
                # Use cached guidance with default guidance scale (typically from config)
                fallback_guidance_fn = self._get_or_create_guidance_fn(self.guidance_scale, batch_guidance_fn, batch_idx)
                for step in range(total_diffusion_steps):
                    batch_plan = self._apply_diffusion_step(batch_plan, conditions, scheduling_matrix, step, pad_tokens, 1, fallback_guidance_fn)
                final_plan[:, batch_idx:batch_idx+1] = batch_plan
        
        # Extract MCTS statistics and best action sequences from all batch elements
        all_mcts_stats = []
        all_best_action_sequences = []
        all_tree_visualizations = []
        
        for batch_idx in range(batch_size):
            root = roots[batch_idx]
            mcts_stats = self._extract_mcts_statistics(root, horizon)
            best_action_sequence = self._extract_best_action_sequence(root, plan_hist[-1][:, batch_idx:batch_idx+1], horizon)
            tree_visualization = self._create_mcts_tree_visualization(root, mcts_stats)
            
            # Display tree structure for each trajectory
            if self.enable_debug_output:
                self._print_mcts_tree_structure(root, batch_idx)
            
            all_mcts_stats.append(mcts_stats)
            all_best_action_sequences.append(best_action_sequence)
            all_tree_visualizations.append(tree_visualization)
        
        # Log aggregated results to wandb
        self._log_mcts_results_batch(all_mcts_stats, all_best_action_sequences, all_tree_visualizations)
        
        plan_hist.append(final_plan.detach()[: self.n_tokens - pad_tokens])

        plan_hist = torch.stack(plan_hist)
        plan_hist = rearrange(plan_hist, "m t b (fs c) -> m (t fs) b c", fs=self.frame_stack)
        plan_hist = plan_hist[:, self.frame_stack : self.frame_stack + horizon]

        # Clear MCTS context flag
        self._in_mcts_context = False

        return plan_hist
    
    def _base_goal_guidance(self, x: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Base goal guidance function without any scaling applied
        Returns raw guidance value that can be scaled by different guidance_scale values
        """
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

        return episode_return

    def _select(self, root: MCTSNode, current_simulation: int = 0, total_simulations: int = 500) -> MCTSNode:
        """Selection phase: traverse tree using UCB1 following Algorithm 3 from MCTD paper"""
        node = root
        # Continue while node is fully expanded AND not a leaf
        while node.is_expanded and not node.is_leaf():
            total_visits = sum(child.visits for child in node.children) + 1
            node = max(node.children, key=lambda c: self._calculate_ucb_score(c, node, total_visits, current_simulation, total_simulations))
        return node
    
    def _select_meta_action(self, node: MCTSNode, current_simulation: int = 0, total_simulations: int = 500) -> Optional[float]:
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
            selected_action = available_actions[0]
            return selected_action
        
        # UCB Selection for multiple available actions
        total_visits = sum(child.visits for child in node.children) + 1  # +1 to avoid log(0)
        
        # Find action with highest UCB score
        best_action = max(available_actions, 
                         key=lambda action: self._calculate_ucb_score(action, node, total_visits, current_simulation, total_simulations))
        
        return best_action

    def _get_or_create_guidance_fn(self, guidance_scale: float, base_guidance_fn, batch_idx: int = 0):
        """Get cached guidance function or create new one for consistency"""
        cache_key = (batch_idx, id(base_guidance_fn), guidance_scale)
        
        if cache_key not in self._guidance_cache:
            if guidance_scale == 0.0 or base_guidance_fn is None:
                # Zero guidance but preserve gradients
                def cached_dummy_guidance(x):
                    return torch.sum(x * 0.0)
                self._guidance_cache[cache_key] = cached_dummy_guidance
            else:
                # Create scaled guidance with proper closure
                def cached_scaled_guidance(x):
                    base_result = base_guidance_fn(x)
                    scaled_result = guidance_scale * base_result

                    return scaled_result
                self._guidance_cache[cache_key] = cached_scaled_guidance
        
        return self._guidance_cache[cache_key]
    
    def _create_guidance_function(self, guidance_scale: float, base_guidance_fn):
        """Legacy method - now uses cached version"""
        return self._get_or_create_guidance_fn(guidance_scale, base_guidance_fn, 0)

    def _calculate_ucb_score(self, action_or_child, node: MCTSNode, total_visits: int, 
                           current_simulation: int = 0, total_simulations: int = 500) -> float:
        """Calculate UCB1 score with reward scaling and parameter annealing"""
        
        if isinstance(action_or_child, MCTSNode):
            # Case 1: Direct child node (for _select method)
            child = action_or_child
            if child.visits == 0:
                return float('inf')
            
            # Step 1: Calculate normalized exploitation term
            exploitation = child.value / child.visits
            
            if total_visits <= 1:  # Avoid log(0) or log(1)
                return exploitation
            
            # Step 2: Get reward range for scaling
            all_nodes = self._get_all_nodes(child.parent) if child.parent else [child]
            reward_values = [n.value / n.visits for n in all_nodes if n.visits > 0]
            
            if len(reward_values) > 1:
                reward_min, reward_max = min(reward_values), max(reward_values)
                reward_range = max(reward_max - reward_min, 1e-6)  # Avoid division by zero
            else:
                reward_range = 1.0  # Default range
            
            # Step 3: Parameter annealing - "先广后精"
            # c(T) decreases as search progresses: c(T) = c_base * (1 - T/T_max)^alpha
            progress = current_simulation / max(total_simulations, 1)
            annealing_factor = (1 - progress) ** self.mcts_annealing_alpha
            c_annealed = self.mcts_c_puct * annealing_factor
            
            # Step 4: Scaled exploration term: c(T) * (b-a) * sqrt(ln(N)/n_i)
            import math
            exploration = c_annealed * reward_range * math.sqrt(math.log(total_visits) / child.visits)
            
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
            
            # Apply same improvements as Case 1
            exploitation = action_value
            
            # Get reward range for scaling
            all_children = node.children if node.children else []
            reward_values = [c.value / c.visits for c in all_children if c.visits > 0]
            
            if len(reward_values) > 1:
                reward_min, reward_max = min(reward_values), max(reward_values)
                reward_range = max(reward_max - reward_min, 1e-6)
            else:
                reward_range = 1.0
            
            # Parameter annealing
            progress = current_simulation / max(total_simulations, 1)
            annealing_factor = (1 - progress) ** self.mcts_annealing_alpha
            c_annealed = self.mcts_c_puct * annealing_factor
            
            # Scaled exploration term
            import math
            exploration = c_annealed * reward_range * math.sqrt(math.log(total_visits) / action_visits)
            
            return exploitation + exploration

    def _denoise_subplan_batch(self, node: MCTSNode, guidance_scale: float, batch_plan: torch.Tensor, 
                              conditions, scheduling_matrix: np.ndarray, start_step: int, 
                              pad_tokens: int, batch_guidance_fn, mcts_subplan_size: int,
                              total_diffusion_steps: int) -> torch.Tensor:
        """Batch-specific DENOISESUBPLAN: Generate new subplan using diffusion for single batch element"""
        # Algorithm 7 Line 4: procedure DENOISESUBPLAN(node, gs)
        temp_plan = batch_plan.clone()
        temp_plan[1 : self.n_tokens - pad_tokens] = node.state
        
        # Algorithm 7 Line 5-9: Create appropriate guidance function (use cached version)
        temp_guidance_fn = self._get_or_create_guidance_fn(guidance_scale, batch_guidance_fn, 0)
        
        # Apply multiple diffusion steps according to mcts_subplan_size
        # This represents a complete "subplan" denoising process
        for step_offset in range(mcts_subplan_size):
            current_step = start_step + step_offset
            if current_step < total_diffusion_steps:
                temp_plan = self._apply_diffusion_step(temp_plan, conditions, scheduling_matrix, 
                                                     current_step, pad_tokens, 1, temp_guidance_fn)
        
        # Return the new subplan state
        return temp_plan[1 : self.n_tokens - pad_tokens]

    def _expand_batch(self, node: MCTSNode, batch_plan: torch.Tensor, conditions, scheduling_matrix: np.ndarray,
                     step: int, pad_tokens: int, batch_guidance_fn, mcts_subplan_size: int,
                     total_diffusion_steps: int, current_simulation: int = 0, total_simulations: int = 500) -> Optional[MCTSNode]:
        """Batch-specific expansion phase for individual batch elements"""
        # Step 2: gs ← SELECTMETAACTION(node) {Determine guidance level}
        guidance_scale = self._select_meta_action(node, current_simulation, total_simulations)
        
        # If no action available (fully expanded), return None
        if guidance_scale is None:
            return None
        
        # Step 3: child ← DENOISESUBPLAN(node, gs) {Generate new subplan using diffusion}
        child_state = self._denoise_subplan_batch(node, guidance_scale, batch_plan, conditions, 
                                                scheduling_matrix, step, pad_tokens, batch_guidance_fn, mcts_subplan_size,
                                                total_diffusion_steps)
        
        # Step 4: ADDCHILD(node, child)
        child = node.add_child(child_state, guidance_scale)
        
        # Mark as fully expanded if all actions have been tried
        if len(node.children) == len(self.guidance_scales):
            node.is_expanded = True
        
        # Step 5: return child
        return child
    
    def _fast_jumpy_denoising_batch(self, node: MCTSNode, batch_plan: torch.Tensor, conditions, 
                                   scheduling_matrix: np.ndarray, pad_tokens: int, 
                                   batch_guidance_fn, mcts_subplan_size: int, 
                                   total_diffusion_steps: int) -> torch.Tensor:
        """Batch-specific FASTJUMPYDENOISING: Complete denoising using finer-grained jumpy interval scale"""
        # Start with the node's current state
        temp_plan = batch_plan.clone()
        temp_plan[1 : self.n_tokens - pad_tokens] = node.state
        
        # Calculate the diffusion step corresponding to this node's depth
        current_depth = self._get_node_depth(node)
        start_step = min(current_depth * mcts_subplan_size, total_diffusion_steps - 1)
        
        # Create guidance function consistent with this node's meta action (use cached version)
        node_guidance_scale = node.action if node.action is not None else 0.0
        consistent_guidance_fn = self._get_or_create_guidance_fn(node_guidance_scale, batch_guidance_fn, 0)
        
        # Calculate new denoising scale based on mcts_jumpy_interval
        # New scale: 1000 timesteps / mcts_jumpy_interval (e.g., 1000/10 = 100 steps)
        new_diffusion_steps = self.timesteps // self.mcts_jumpy_interval
        
        # Map current progress from old scale to new scale
        # start_step is in old scale (0 to total_diffusion_steps-1)
        # Convert to new scale (0 to new_diffusion_steps-1)
        progress_ratio = start_step / total_diffusion_steps
        new_start_step = int(progress_ratio * new_diffusion_steps)
        
        # Construct new scheduling matrix based on the new denoising scale
        # Preserve the independent noise level mechanism for each token
        new_scheduling_matrix = self._construct_jumpy_scheduling_matrix(
            scheduling_matrix, new_diffusion_steps, self.mcts_jumpy_interval
        )
        
        # Apply remaining diffusion steps using the new finer-grained scale
        for step in range(new_start_step, new_diffusion_steps):
            # Use the new scheduling matrix to maintain independent noise levels
            temp_plan = self._apply_diffusion_step(
                temp_plan, conditions, new_scheduling_matrix, 
                step, pad_tokens, 1, consistent_guidance_fn
            )
        
        return temp_plan
    
    def _construct_jumpy_scheduling_matrix(self, original_scheduling_matrix: np.ndarray, 
                                         new_diffusion_steps: int, jumpy_interval: int) -> np.ndarray:
        """Construct new scheduling matrix based on jumpy interval while preserving independent noise levels"""
        # Get the original matrix dimensions
        original_steps, num_tokens = original_scheduling_matrix.shape
        
        # Create new scheduling matrix
        new_scheduling_matrix = np.zeros((new_diffusion_steps + 1, num_tokens), dtype=np.int64)
        
        # For each step in the new scale, interpolate from the original scheduling matrix
        for new_step in range(new_diffusion_steps + 1):
            # Calculate corresponding timestep in original 1000-step scale
            timestep_in_original_scale = (new_diffusion_steps - new_step) * jumpy_interval
            
            # Map this timestep to the original scheduling matrix
            # The original matrix maps from 1000 steps to ~20 steps
            # We need to find which row in original matrix corresponds to this timestep
            original_progress = 1.0 - (timestep_in_original_scale / self.timesteps)  # 0 to 1
            original_step = int(original_progress * (original_steps - 1))
            original_step = max(0, min(original_step, original_steps - 1))
            
            # Copy the noise levels from the original matrix
            new_scheduling_matrix[new_step] = original_scheduling_matrix[original_step]
        
        return new_scheduling_matrix
    
    def _parse_manual_path_format(self, manual_path) -> List[float]:
        """
        Parse manual path format that supports both:
        1. Simple list format: [2, 2, 2] 
        2. Path string format: "Path: (0, ROOT) -> (1, 2) -> (2, 0) -> (3, 0) -> (4, 2)"
        """
        if isinstance(manual_path, list):
            # Simple list format - return as is
            return manual_path
        elif isinstance(manual_path, str) and manual_path.startswith("Path:"):
            # Parse path string format
            import re
            # Extract (depth, action) pairs from the string
            pattern = r'\((\d+),\s*(\w+|[\d.]+)\)'
            matches = re.findall(pattern, manual_path)
            
            guidance_scales = []
            for depth_str, action_str in matches:
                depth = int(depth_str)
                if action_str == "ROOT":
                    # Skip root node - it will be added automatically
                    continue
                else:
                    # Convert action to float guidance scale
                    try:
                        guidance_scale = float(action_str)
                        guidance_scales.append(guidance_scale)
                    except ValueError:
                        print(f"Warning: Cannot parse action '{action_str}' as guidance scale, skipping")
            
            
            return guidance_scales
        else:
            # Fallback to treating as simple list
            return manual_path if isinstance(manual_path, list) else [manual_path]

    def _create_manual_optimal_path(self, manual_path, dummy_state: torch.Tensor) -> List[MCTSNode]:
        """Create a manual optimal path from parsed guidance scales"""
        # Parse the manual path format (supports both list and string formats)
        guidance_scales = self._parse_manual_path_format(manual_path)
        
        path = []
        
        # Create root node with no action (consistent with MCTS structure)
        root = MCTSNode(state=dummy_state, parent=None, action=None)
        root.visits = 1
        root.value = 1.0
        path.append(root)
        current_node = root
        
        # Create child nodes with specified guidance scales
        for depth, guidance_scale in enumerate(guidance_scales):
            # Create a node with the specified guidance scale
            # Use dummy_state as placeholder since we won't use the actual state
            node = MCTSNode(state=dummy_state, parent=current_node, action=guidance_scale)
            node.visits = 1  # Set minimal visits to avoid division by zero
            node.value = 1.0  # Set positive value to indicate good performance
            
            current_node.children.append(node)
            path.append(node)
            current_node = node
        
        return path
    
    def _simulate_interact_for_goal_reward_batch(self, full_plan: torch.Tensor, batch_guidance_fn, horizon: int, node_guidance_scale: Optional[float], batch_goal: torch.Tensor) -> float:
        """
        Batch-specific simplified simulation of interact() logic for goal reward calculation
        """
        # Convert full_plan to trajectory format
        plan_traj = rearrange(full_plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        plan_traj = plan_traj[self.frame_stack:]  # Remove initial padding
        
        # Extract batch size and limit horizon (should be 1 for batch-specific)
        batch_size = plan_traj.shape[1]
        actual_horizon = min(horizon, plan_traj.shape[0])
        
        # Use node-specific guidance scale if provided, otherwise fallback to global guidance scale
        effective_guidance_scale = node_guidance_scale if node_guidance_scale is not None else self.guidance_scale
        
        
        # Initialize tracking variables like in interact()
        reached = torch.zeros(batch_size, dtype=torch.bool, device=full_plan.device)
        first_reach = torch.zeros(batch_size, device=full_plan.device)
        
        # Extract goal position (should be single batch element)
        goal_position = batch_goal[:, :2]  # Shape: (1, 2)
        
        # Track minimum position distance for debug output
        min_position_dist = float('inf')
        
        # Simulate step-by-step trajectory execution like interact()
        for t in range(actual_horizon):
            # Extract current position from the trajectory
            current_obs, _, _ = self.split_bundle(plan_traj[t])  # Shape: (1, obs_dim)
            current_position = current_obs[:, :2]  # Assume first 2 dims are x, y coordinates

            # Calculate position-only guidance directly using current_position
            position_dist = torch.norm(current_position - goal_position, dim=-1)  # (1,) - Euclidean distance for time step t
            
            # Update minimum position distance
            min_position_dist = min(min_position_dist, position_dist.min().item())

            newly_reached = (position_dist <= 0.5) & (~reached)  # Corresponds to distance <= 0.45

            reached = reached | newly_reached
            # Update first_reach counter (increment for non-reached samples)
            first_reach += (~reached).float()
        
        # Final debug output
        if self.enable_debug_output:
            print(f"🔍 Batch Final Results - Min position dist: {min_position_dist:.6f}, Reached any: {reached.any().item()}, First reach mean: {first_reach.mean().item():.2f}")
        
        # Return first_reach.mean() equivalent as used in interact()
        return first_reach.mean().item()
    

    def _evaluate_plan_batch(self, full_plan: torch.Tensor, batch_guidance_fn, node: MCTSNode, batch_goal: torch.Tensor) -> float:
        """
        Batch-specific EVALUATEPLAN: Evaluate the quality of a complete plan for single batch element
        """
        # Debug output: Trace path from root to current node
        if self.enable_debug_output:
            path = []
            current = node
            # Trace back to root
            while current is not None:
                depth = self._get_node_depth(current)
                action = current.action if current.action is not None else "ROOT"
                path.append((depth, action))
                current = current.parent
            
            # Reverse to show root -> ... -> current
            path.reverse()
            
            # Format path as requested: (depth, meta action) -> (depth, meta action)
            path_str = " -> ".join([f"({depth}, {action})" for depth, action in path])
            print(f"Batch Path: {path_str}")
        
        # Convert full_plan to trajectory format for evaluation
        plan_traj = rearrange(full_plan, "t b (fs c) -> (t fs) b c", fs=self.frame_stack)
        plan_traj = plan_traj[self.frame_stack:]  # Remove initial padding
        
        # Extract observations from the plan
        observations, _, _ = self.split_bundle(plan_traj)
        batch_size = observations.shape[1]  # Should be 1 for batch-specific
        horizon = observations.shape[0]
        
        total_reward = 0.0
        
        # Rule 1: Check for physically impossible large position differences between near states
        position_penalty = 0.0
        if horizon > 1:
            # Extract positions (assuming first 2 dimensions are x, y coordinates)
            positions = observations[:, :, :2]  # Shape: (T, 1, 2)
            
            # Calculate position differences between consecutive states
            pos_diffs = torch.diff(positions, dim=0)  # Shape: (T-1, 1, 2)
            pos_distances = torch.norm(pos_diffs, dim=2)  # Shape: (T-1, 1)
            
            # Define maximum physically reasonable distance per step
            max_step_distance = 0.10  # Adjust based on environment specifics
            
            # Penalty for unrealistic jumps - normalized by total steps
            large_jumps = pos_distances > max_step_distance
            total_steps = pos_distances.shape[0]  # T-1 steps
            position_penalty = -large_jumps.float().sum().item() / max(total_steps, 1) * 0.5  # Normalized penalty
        
        # Rule 2: Reward for reaching the goal using first_reach metric borrowed from interact()
        goal_reward = 0.0
        if horizon > 0:
            # Use simplified interact() logic for goal reward calculation
            # Pass the node's guidance scale to ensure proper evaluation
            node_guidance_scale = node.action if node.action is not None else 0.0
            first_reach_mean = self._simulate_interact_for_goal_reward_batch(full_plan, batch_guidance_fn, horizon, node_guidance_scale, batch_goal)
            
            # Calculate goal reward using first_reach metric with r = (H - t)/H formula
            # first_reach_mean represents the average time to reach goal (or horizon if not reached)
            if first_reach_mean < horizon:
                # Goal was reached, calculate early reaching reward
                t_reach = first_reach_mean  # Average first reach time
                goal_reward = (horizon - t_reach) / horizon  # r = (H - t)/H
            else:
                # No goal reached, minimal reward
                goal_reward = 0.0

        if self.enable_debug_output:
            print(f"Batch Position Penalty: {position_penalty}")  
            print(f"Batch Goal Reward: {goal_reward}")
        # Combine all reward components
        total_reward = position_penalty * 0 + goal_reward
        
        return total_reward
    
    def _simulate_batch(self, node: MCTSNode, batch_plan: torch.Tensor, conditions, 
                       scheduling_matrix: np.ndarray, pad_tokens: int, 
                       batch_guidance_fn, mcts_subplan_size: int, total_diffusion_steps: int, 
                       batch_start: torch.Tensor, batch_goal: torch.Tensor) -> float:
        """Batch-specific simulation phase for individual batch elements"""
        current_depth = self._get_node_depth(node)
        
        if self.enable_debug_output and current_depth >= 2:  # Debug deep paths
            path_trace = []
            temp_node = node
            while temp_node is not None:
                action = temp_node.action if temp_node.action is not None else "ROOT"
                path_trace.append(f"({self._get_node_depth(temp_node)}, {action})")
                temp_node = temp_node.parent
            path_trace.reverse()
            path_str = " -> ".join(path_trace)
            print(f"🔍 Simulating Path: {path_str}, Depth: {current_depth}/{self.mcts_depth}")
        
        if current_depth >= self.mcts_depth:
            # Full path: node already represents complete denoising, no Fast Jumpy needed
            temp_plan = batch_plan.clone()
            temp_plan[1 : self.n_tokens - pad_tokens] = node.state
            full_plan = temp_plan
        else:
            # Partial path: use Fast Jumpy to complete remaining denoising
            full_plan = self._fast_jumpy_denoising_batch(node, batch_plan, conditions, scheduling_matrix, 
                                                       pad_tokens, batch_guidance_fn, mcts_subplan_size, 
                                                       total_diffusion_steps)
        
        # Step 3: return EVALUATEPLAN(fullPlan)
        return self._evaluate_plan_batch(full_plan, batch_guidance_fn, node, batch_goal)
    
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
                
                # Node label with (depth, meta action) tuple, visits and value info
                if node.visits > 0:
                    avg_value = node.value / node.visits
                    if hasattr(node, 'action') and node.action is not None:
                        node_labels[current_id] = f"({depth}, {node.action})\nV:{node.visits}\nR:{avg_value:.2f}"
                    else:
                        node_labels[current_id] = f"({depth}, ROOT)\nV:{node.visits}\nR:{avg_value:.2f}"
                else:
                    if hasattr(node, 'action') and node.action is not None:
                        node_labels[current_id] = f"({depth}, {node.action})\nV:0\nR:0.0"
                    else:
                        node_labels[current_id] = f"({depth}, ROOT)\nV:0\nR:0.0"
                
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
    
    def _apply_optimal_path_batch(self, batch_plan: torch.Tensor, optimal_path: List[MCTSNode], 
                                 conditions, scheduling_matrix: np.ndarray, pad_tokens: int, 
                                 batch_guidance_fn, total_diffusion_steps: int, batch_idx: int = 0) -> torch.Tensor:
        """Optimized batch-specific apply optimal path using step mapping and cached guidance"""
        current_plan = batch_plan.clone()
        
        effective_path_length = len(optimal_path) - 1  # Exclude root node
        steps_per_level = max(1, total_diffusion_steps // effective_path_length) if effective_path_length > 0 else total_diffusion_steps
        
        # Pre-compute step to guidance_scale mapping
        step_to_scale = {}
        current_step = 0
        for i, node in enumerate(optimal_path[1:]):  # Skip root node
            guidance_scale = node.action if hasattr(node, 'action') and node.action is not None else 0.0
            level_steps = min(steps_per_level, total_diffusion_steps - current_step)
            
            for step_offset in range(level_steps):
                if current_step < total_diffusion_steps:
                    step_to_scale[current_step] = guidance_scale
                    current_step += 1
        
        # Fill remaining steps with final guidance scale
        if optimal_path and current_step < total_diffusion_steps:
            final_node = optimal_path[-1]
            final_guidance_scale = final_node.action if hasattr(final_node, 'action') and final_node.action is not None else 0.0
            for remaining_step in range(current_step, total_diffusion_steps):
                step_to_scale[remaining_step] = final_guidance_scale
        
        # Apply diffusion steps using cached guidance functions
        
        for step in range(total_diffusion_steps):
            guidance_scale = step_to_scale.get(step, 0.0)
            
            # Use cached guidance function for consistency with MCTS search phase
            step_guidance_fn = self._get_or_create_guidance_fn(guidance_scale, batch_guidance_fn, batch_idx)
            
            current_plan = self._apply_diffusion_step(
                current_plan, conditions, scheduling_matrix, 
                step, pad_tokens, 1, step_guidance_fn
            )
        
        return current_plan
    
    def _print_mcts_tree_structure(self, root: MCTSNode, batch_idx: int):
        """Print MCTS tree structure in console format showing (depth, meta action) tuples with visits and values"""
        print(f"\n🌳 MCTS Tree Structure - Sample_{batch_idx}")
        print("=" * 60)
        
        def print_node_recursive(node: MCTSNode, depth: int = 0, is_last: bool = True, prefix: str = ""):
            # Prepare node info
            if hasattr(node, 'action') and node.action is not None:
                action_str = str(node.action)
            else:
                action_str = "ROOT"
            
            # Calculate average value
            avg_value = node.value / node.visits if node.visits > 0 else 0.0
            
            # Format node label
            node_label = f"({depth}, {action_str}) - V:{node.visits}, R:{avg_value:.2f}"
            
            # Print with tree structure
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{node_label}")
            
            # Update prefix for children
            if node.children:
                child_prefix = prefix + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    is_last_child = (i == len(node.children) - 1)
                    print_node_recursive(child, depth + 1, is_last_child, child_prefix)
        
        # Start recursive printing from root
        print_node_recursive(root)
        
        # Print summary statistics
        total_nodes = len(list(self._get_all_nodes(root)))
        total_visits = sum(node.visits for node in self._get_all_nodes(root))
        max_depth = max(self._get_node_depth(node) for node in self._get_all_nodes(root))
        
        print(f"\nTree Statistics:")
        print(f"  Total Nodes: {total_nodes}")
        print(f"  Total Visits: {total_visits}")
        print(f"  Max Depth: {max_depth}")
        
        # Print optimal trajectory path
        print(f"\n🎯 Optimal Trajectory Path - Sample_{batch_idx}:")
        optimal_path = self._extract_optimal_path(root)
        path_str = " -> ".join([
            f"({self._get_node_depth(node)}, {node.action if hasattr(node, 'action') and node.action is not None else 'ROOT'})"
            for node in optimal_path
        ])
        print(f"  Path: {path_str}")
        
        # Print detailed path with statistics
        print(f"  Detailed Path:")
        for i, node in enumerate(optimal_path):
            depth = self._get_node_depth(node)
            action = node.action if hasattr(node, 'action') and node.action is not None else 'ROOT'
            avg_value = node.value / node.visits if node.visits > 0 else 0.0
            
            if i == 0:
                print(f"    Start: ({depth}, {action}) - V:{node.visits}, R:{avg_value:.2f}")
            elif i == len(optimal_path) - 1:
                print(f"    End:   ({depth}, {action}) - V:{node.visits}, R:{avg_value:.2f}")
            else:
                print(f"    Step:  ({depth}, {action}) - V:{node.visits}, R:{avg_value:.2f}")
        
        print("=" * 60)
    
    def _log_mcts_results_batch(self, all_mcts_stats: List[Dict[str, Any]], all_best_action_sequences: List[Dict[str, Any]], all_tree_visualizations: List[Optional[Any]]):
        """Log aggregated MCTS results from all batch elements"""
        import numpy as np
        
        batch_size = len(all_mcts_stats)
        
        # Aggregate statistics across all batch elements
        total_nodes_all = [stats['total_nodes'] for stats in all_mcts_stats]
        total_visits_all = [stats['total_visits'] for stats in all_mcts_stats]
        max_depth_all = [stats['max_depth'] for stats in all_mcts_stats]
        
        # Log aggregated statistics
        self.log("mcts_batch/avg_total_nodes", float(np.mean(total_nodes_all)))
        self.log("mcts_batch/avg_total_visits", float(np.mean(total_visits_all)))
        self.log("mcts_batch/avg_max_depth", float(np.mean(max_depth_all)))
        self.log("mcts_batch/std_total_nodes", float(np.std(total_nodes_all)))
        self.log("mcts_batch/std_total_visits", float(np.std(total_visits_all)))
        self.log("mcts_batch/std_max_depth", float(np.std(max_depth_all)))
        
        # Log best actions across batch
        best_actions_all = [stats.get('best_action', 0) for stats in all_mcts_stats]
        self.log("mcts_batch/avg_best_guidance_scale", float(np.mean(best_actions_all)))
        self.log("mcts_batch/std_best_guidance_scale", float(np.std(best_actions_all)))
        
        # Log plan length statistics
        plan_lengths_all = [seq['plan_length'] for seq in all_best_action_sequences]
        self.log("mcts_batch/avg_plan_length", float(np.mean(plan_lengths_all)))
        self.log("mcts_batch/std_plan_length", float(np.std(plan_lengths_all)))
        
        # Log individual batch element results for detailed analysis
        for batch_idx, (mcts_stats, action_sequence, tree_viz) in enumerate(zip(all_mcts_stats, all_best_action_sequences, all_tree_visualizations)):
            # Log per-batch metrics
            self.log(f"mcts_batch/batch_{batch_idx}_total_nodes", mcts_stats['total_nodes'])
            self.log(f"mcts_batch/batch_{batch_idx}_total_visits", mcts_stats['total_visits'])
            self.log(f"mcts_batch/batch_{batch_idx}_max_depth", mcts_stats['max_depth'])
            
            if 'best_action' in mcts_stats:
                self.log(f"mcts_batch/batch_{batch_idx}_best_guidance_scale", mcts_stats['best_action'])
            
            # Log tree visualization if available
            if tree_viz is not None:
                self.log_image(f"mcts_batch/batch_{batch_idx}_tree_structure", tree_viz)
        
        # Log a summary tree visualization (use first batch element as representative)
        if all_tree_visualizations and all_tree_visualizations[0] is not None:
            self.log_image("mcts_batch/representative_tree_structure", all_tree_visualizations[0])

    def _apply_diffusion_step(self, plan: torch.Tensor, conditions, scheduling_matrix: np.ndarray,
                             step: int, pad_tokens: int, batch_size: int, guidance_fn) -> torch.Tensor:
        """Apply a single diffusion denoising step"""
        # Get actual batch size from plan tensor to match df_planning behavior
        actual_batch_size = plan.shape[1] if len(plan.shape) > 1 else batch_size
        
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
        from_noise_levels = repeat(from_noise_levels, "t -> t b", b=actual_batch_size)
        to_noise_levels = repeat(to_noise_levels, "t -> t b", b=actual_batch_size)
        
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