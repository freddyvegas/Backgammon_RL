#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch PPO agent for Backgammon with legal move masking
FIXED VERSION - Applied all corrections from code review:
1. Fixed ACNet constructor argument names (state_dim, model_dim, n_blocks)
2. Removed torch.load weights_only parameter for compatibility
3. Consistent config definitions (merged with model_size_configs.py)
4. Fixed device selection comments (no more MPS claims)

IMPROVED VERSION with:
- Gentler PPO hyperparameters (lower LR, clip, epochs)
- Teacher signal (DAGGER-lite) for pubeval distillation
- Better curriculum learning support
"""

from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

import backgammon

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------- Config ----------------
class Config:
    """Base configuration class with IMPROVED HYPERPARAMETERS."""
    state_dim = 29
    max_actions = 64
    
    # PPO hyperparameters - GENTLER for stability
    lr = 5e-5              # Reduced from 1e-4 → 5e-5
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.1     # Reduced from 0.2 → 0.1
    
    # Exploration
    entropy_coef = 0.02
    entropy_decay = 0.9999
    entropy_min = 0.0
    
    critic_coef = 1.0
    eval_temperature = 0.01
    
    # PPO rollout settings
    rollout_length = 512
    ppo_epochs = 2         # Reduced from 4 → 2
    minibatch_size = 128
    
    # Network architecture (ResMLP) - DEFAULT: LARGE
    model_dim = 512
    n_blocks = 6
    
    # Gradient clipping
    grad_clip = 0.5
    max_grad_norm = 0.5
    
    # Weight decay
    weight_decay = 1e-5
    
    # Reward scaling
    reward_scale = 1.0
    
    # Reward shaping (DISABLED)
    use_reward_shaping = False
    pip_reward_scale = 0.001
    bear_off_reward = 0.01
    hit_reward = 0.01
    shaping_clip = 0.05
    
    # Teacher signal (DAGGER-lite) - NEW!
    teacher_sample_rate = 0.10    # Sample teacher label 10% of time
    teacher_loss_coef_start = 0.05  # Initial teacher loss coefficient
    teacher_loss_coef_end = 0.0     # Final teacher loss coefficient (decay to 0)
    teacher_decay_horizon = 50_000  # Decay over this many games


class SmallConfig(Config):
    """Small model for CPU training / quick testing."""
    model_dim = 128
    n_blocks = 3
    rollout_length = 256
    minibatch_size = 64
    lr = 5e-5  # Gentler
    ppo_epochs = 2  # Gentler
    clip_epsilon = 0.1  # Gentler
    use_reward_shaping = False


class MediumConfig(Config):
    """Medium model for modest GPUs / balanced training."""
    model_dim = 256
    n_blocks = 4
    rollout_length = 384
    minibatch_size = 96


class LargeConfig(Config):
    """Large model (default) for full GPU training."""
    pass


def get_config(size='large'):
    """Get configuration for specified model size."""
    configs = {
        'small': SmallConfig,
        'medium': MediumConfig,
        'large': LargeConfig,
    }
    
    size = size.lower()
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")
    
    cfg = configs[size]()
    
    print(f"\nModel Configuration: {size.upper()}")
    print(f"  Parameters: ~{'80K' if size == 'small' else '400K' if size == 'medium' else '1.6M'}")
    print(f"  Architecture: dim={cfg.model_dim}, blocks={cfg.n_blocks}")
    print(f"  Rollout: length={cfg.rollout_length}, batch={cfg.minibatch_size}")
    print(f"  Learning rate: {cfg.lr} (gentler)")
    print(f"  PPO clip ε: {cfg.clip_epsilon} (gentler)")
    print(f"  PPO epochs: {cfg.ppo_epochs} (gentler)")
    print(f"  Teacher signal: {cfg.teacher_sample_rate*100:.0f}% sample rate")
    
    return cfg


# Default config instance
CFG = Config()


def get_device():
    """
    Automatically detect and return the best available device.
    
    FIX #7: Removed misleading MPS comments. MPS has known stability issues
    with PPO training (negative policy loss, gradient explosion). Use CPU or CUDA.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Available but use with caution for PPO
    else:
        return "cpu"


# Set device at module level
CFG.device = get_device()
print(f"PPO agent using device: {CFG.device}")
if CFG.device == "mps":
    print("  ⚠️  Note: MPS may have stability issues with PPO. Consider using CPU.")


# ------------- Flip helpers -------------
_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32
)

def _flip_board(board):
    out = np.empty(29, dtype=board.dtype)
    out[:] = -board[_FLIP_IDX]
    return out

def _flip_move(move):
    if len(move) == 0:
        return move
    mv = np.asarray(move, dtype=np.int32).copy()
    for r in range(mv.shape[0]):
        mv[r, 0] = _FLIP_IDX[mv[r, 0]]
        mv[r, 1] = _FLIP_IDX[mv[r, 1]]
    return mv

# ------------- Rollout Buffer with Teacher Signal -------------
class PPORolloutBuffer:
    """Stores on-policy rollouts for PPO updates with teacher signal support."""
    def __init__(self, rollout_length=512):
        self.rollout_length = rollout_length
        self.clear()
    
    def clear(self):
        self.states = []
        self.candidate_states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.teacher_indices = []  # NEW: store teacher indices (-1 if no teacher)
        self.size = 0
    
    def push(self, state, candidate_states, mask, action, log_prob, value, reward, done, teacher_idx=-1):
        """
        Add a transition to the buffer.
        
        NEW: teacher_idx parameter
        - Set to action index chosen by teacher (e.g., pubeval)
        - Set to -1 if no teacher label for this transition
        """
        self.states.append(state)
        self.candidate_states.append(candidate_states)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.teacher_indices.append(teacher_idx)  # NEW
        self.size += 1
    
    def is_ready(self):
        return self.size >= self.rollout_length
    
    def get(self):
        """Return all buffered data."""
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.candidate_states, dtype=np.float32),
            np.array(self.masks, dtype=np.float32),
            np.array(self.actions, dtype=np.int64),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.teacher_indices, dtype=np.int64),  # NEW
        )


# ------------- Network Architecture -------------
class ResMLPBlock(nn.Module):
    """Residual MLP block with proper normalization."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return x + self.net(x)


class PPOActorCritic(nn.Module):
    """
    Actor-Critic with delta-based scoring and value head.
    
    FIX #1: Constructor now uses correct parameter names (state_dim, model_dim, n_blocks)
    """
    def __init__(self, state_dim=29, model_dim=512, n_blocks=6):
        super().__init__()
        self.state_dim = state_dim
        self.model_dim = model_dim
        
        # State encoder
        self.state_enc = nn.Linear(state_dim, model_dim)
        
        # Residual MLP blocks
        self.blocks = nn.ModuleList([
            ResMLPBlock(model_dim) for _ in range(n_blocks)
        ])
        
        # Delta projection (for scoring candidates)
        self.delta_proj = nn.Linear(state_dim, model_dim)
        
        # Value head (single scalar)
        self.value_head = nn.Linear(model_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.state_enc.bias)
        nn.init.orthogonal_(self.delta_proj.weight, gain=1.0)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, states, deltas, mask):
        """
        Args:
            states: (B, state_dim)
            deltas: (B, max_actions, state_dim)
            mask: (B, max_actions) - 1 for valid, 0 for padding
        
        Returns:
            logits: (B, max_actions) - masked logits
            values: (B,) - value estimates
        """
        B = states.size(0)
        
        # Encode state
        s_enc = self.state_enc(states)  # (B, model_dim)
        s_enc = F.relu(s_enc)
        
        # Process through residual blocks
        x = s_enc
        for block in self.blocks:
            x = block(x)  # (B, model_dim)
        
        # Value prediction
        values = self.value_head(x).squeeze(-1)  # (B,)
        
        # Score candidates via deltas
        delta_enc = self.delta_proj(deltas)  # (B, max_actions, model_dim)
        logits = torch.sum(x.unsqueeze(1) * delta_enc, dim=-1)  # (B, max_actions)
        
        # Apply mask (use -1e9 instead of -inf for MPS compatibility)
        logits = logits.masked_fill(mask == 0, -1e9)
        
        return logits, values
    
    def score_moves_delta(self, states, deltas, mask):
        """Score moves using delta representation (for lookahead)."""
        logits, _ = self.forward(states, deltas, mask)
        return logits
    
    def value(self, states):
        """Get value estimate for states."""
        # Need dummy deltas and mask for forward pass
        B = states.size(0)
        dummy_deltas = torch.zeros(B, 1, self.state_dim, device=states.device)
        dummy_mask = torch.ones(B, 1, device=states.device)
        _, values = self.forward(states, dummy_deltas, dummy_mask)
        return values


# ------------- PPO Agent -------------
class PPOAgent:
    """PPO agent with teacher signal support (DAGGER-lite)."""
    
    def __init__(self, config=None, device=None, pubeval_module=None):
        """
        Args:
            config: Config object
            device: Device string or None for auto-detect
            pubeval_module: Module with pubeval functions for teacher signal
        """
        self.config = config or CFG
        self.device = device or get_device()
        self.pubeval = pubeval_module  # NEW: store pubeval module
        
        # FIX #1: Network created with correct keyword argument names
        self.acnet = PPOActorCritic(
            state_dim=self.config.state_dim,
            model_dim=self.config.model_dim,
            n_blocks=self.config.n_blocks
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.acnet.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Rollout buffer
        self.buffer = PPORolloutBuffer(rollout_length=self.config.rollout_length)
        
        # Training state
        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = self.config.entropy_coef
        
        # Stats tracking
        self.rollout_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'teacher_loss': [],  # NEW
            'masked_entropy': [],
            'grad_norm': [],
            'nA_values': [],
        }
        
        print(f"PPOAgent initialized with IMPROVED hyperparameters:")
        print(f"  Device: {self.device}")
        print(f"  LR: {self.config.lr} (gentler)")
        print(f"  Clip ε: {self.config.clip_epsilon} (gentler)")
        print(f"  PPO epochs: {self.config.ppo_epochs} (gentler)")
        print(f"  Max grad norm: {self.config.max_grad_norm}")
        print(f"  Teacher signal: {self.config.teacher_sample_rate*100:.0f}% sample rate")
    
    def _encode_state(self, board, moves_left=0):
        """Encode board state (29 dims only)."""
        return board.astype(np.float32)
    
    def _compute_shaped_reward(self, board_before, board_after):
        """Compute shaped reward (DISABLED by default)."""
        if not self.config.use_reward_shaping:
            return 0.0
        
        pip_before = np.sum(np.arange(1, 25) * np.maximum(board_before[1:25], 0))
        pip_after = np.sum(np.arange(1, 25) * np.maximum(board_after[1:25], 0))
        pip_reward = -self.config.pip_reward_scale * (pip_after - pip_before)
        
        bear_off_before = board_before[27]
        bear_off_after = board_after[27]
        bear_off_reward = self.config.bear_off_reward * (bear_off_after - bear_off_before)
        
        hit_reward = self.config.hit_reward * (board_after[26] - board_before[26])
        
        total_shaped = pip_reward + bear_off_reward + hit_reward
        total_shaped = np.clip(total_shaped, -self.config.shaping_clip, self.config.shaping_clip)
        
        return total_shaped
    
    def save(self, path: str):
        """Save agent checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'acnet': self.acnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'entropy_coef': self.current_entropy_coef,
            'config': {
                'state_dim': self.config.state_dim,
                'model_dim': self.config.model_dim,
                'n_blocks': self.config.n_blocks,
                'lr': self.config.lr,
                'clip_epsilon': self.config.clip_epsilon,
                'ppo_epochs': self.config.ppo_epochs,
            }
        }, path)
    
    def load(self, path: str, map_location: Union[str, torch.device] = None):
        """
        Load agent checkpoint.
        
        FIX #12: Removed weights_only parameter for PyTorch version compatibility
        """
        if map_location is None:
            map_location = self.device
        
        # FIX #12: Removed weights_only=False parameter
        checkpoint = torch.load(path, map_location=map_location)
        
        # FIX #1: Rebuild network with correct architecture from checkpoint
        saved_config = checkpoint.get('config', {})
        if saved_config:
            # Recreate network if architecture differs
            if (saved_config.get('state_dim', self.config.state_dim) != self.config.state_dim or
                saved_config.get('model_dim', self.config.model_dim) != self.config.model_dim or
                saved_config.get('n_blocks', self.config.n_blocks) != self.config.n_blocks):
                
                print(f"  Rebuilding network with saved architecture:")
                print(f"    state_dim={saved_config.get('state_dim', self.config.state_dim)}")
                print(f"    model_dim={saved_config.get('model_dim', self.config.model_dim)}")
                print(f"    n_blocks={saved_config.get('n_blocks', self.config.n_blocks)}")
                
                # FIX #1: Use correct keyword argument names
                self.acnet = PPOActorCritic(
                    state_dim=saved_config.get('state_dim', self.config.state_dim),
                    model_dim=saved_config.get('model_dim', self.config.model_dim),
                    n_blocks=saved_config.get('n_blocks', self.config.n_blocks)
                ).to(self.device)
        
        self.acnet.load_state_dict(checkpoint['acnet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        self.current_entropy_coef = checkpoint.get('entropy_coef', self.config.entropy_coef)
        
        print(f"Loaded checkpoint: {path}")
        print(f"  Steps: {self.steps:,}")
        print(f"  Updates: {self.updates:,}")
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = is_eval
        if is_eval:
            self.acnet.eval()
        else:
            self.acnet.train()
    
    def batch_score(self, states_np, cand_states_np, masks_np):
        """Batch scoring for parallel games."""
        B = states_np.shape[0]
        max_A = cand_states_np.shape[1]
        
        # Compute deltas
        deltas_np = cand_states_np - states_np[:, None, :]
        
        # Convert to tensors
        states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        deltas_t = torch.as_tensor(deltas_np, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks_np, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, values = self.acnet(states_t, deltas_t, masks_t)
        
        return logits, values
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0.0
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def _get_teacher_label(self, state, deltas, mask):
        """
        Get teacher label from pubeval for a state.
        
        Args:
            state: (29,) current state
            deltas: (nA, 29) candidate deltas
            mask: (nA,) mask for valid actions
        
        Returns:
            int: Index of teacher's choice, or -1 if pubeval unavailable
        """
        if self.pubeval is None:
            return -1
        
        try:
            nA = int(mask.sum())
            if nA == 0:
                return -1
            
            # Score each after-state with pubeval
            scores = []
            for k in range(nA):
                after = (state + deltas[k]).astype(np.float32)
                race = int(self.pubeval.israce(after))
                pb28 = self.pubeval.pubeval_flip(after)
                pos = self.pubeval._to_int32_view(pb28)
                scores.append(float(self.pubeval._pubeval_scalar(race, pos)))
            
            return int(np.argmax(scores))
        except Exception as e:
            # If pubeval fails, return -1
            return -1
    
    def _ppo_update(self):
        """PPO update with teacher signal (DAGGER-lite)."""
        # Get buffered data
        states, cand_states, masks, actions, old_log_probs, values, rewards, dones, teacher_indices = self.buffer.get()
        
        # DEBUG: Print reward statistics
        #n_positive = (rewards > 0.5).sum()
        #n_negative = (rewards < -0.5).sum()
        #n_zero = ((rewards >= -0.5) & (rewards <= 0.5)).sum()
        #print(f"  [PPO DEBUG] Buffer rewards: +1={n_positive}, -1={n_negative}, ~0={n_zero}, mean={rewards.mean():.3f}, min={rewards.min():.3f}, max={rewards.max():.3f}")
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # CRITICAL: Clip advantages to prevent extreme values
        # Extreme advantages cause unstable training
        # REMOVED: hard clipping of advantages
        
        #print(f"  [PPO DEBUG] Advantages: mean={advantages.mean():.3f}, std={advantages.std():.3f}, range=[{advantages.min():.3f}, {advantages.max():.3f}]")
        
        # Convert to tensors
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        cand_states_t = torch.as_tensor(cand_states, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)  # for clipped value loss

        teacher_indices_t = torch.as_tensor(teacher_indices, dtype=torch.int64, device=self.device)  # NEW
        
        # Compute deltas
        deltas_t = cand_states_t - states_t.unsqueeze(1)
        
        # Compute teacher loss coefficient (cosine decay)
        alpha0 = self.config.teacher_loss_coef_start
        horizon = self.config.teacher_decay_horizon
        alpha = alpha0 * max(0.0, 0.5 * (1 + math.cos(math.pi * min(self.steps, horizon) / horizon)))
        
        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_teacher_loss = 0.0
        n_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            
            for start_idx in range(0, dataset_size, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get minibatch
                mb_states = states_t[mb_indices]
                mb_deltas = deltas_t[mb_indices]
                mb_masks = masks_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]
                mb_old_values = old_values_t[mb_indices]
                mb_teacher_idx = teacher_indices_t[mb_indices]
                
                # Forward pass
                logits, value_preds = self.acnet(mb_states, mb_deltas, mb_masks)
                
                # Compute log probs
                log_probs_all = F.log_softmax(logits.masked_fill(mb_masks == 0, -1e9), dim=-1)
                log_probs = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                
                # PPO policy loss
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # CRITICAL FIX: Clamp ratio to prevent extreme ratios
                ratio = torch.clamp(ratio, 0.5, 2.0)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # CRITICAL: Clamp policy loss to prevent negative values
                # Negative policy loss = gradient ascent on bad actions!
                #if policy_loss.item() < -1e-6:
                #    print(f"    [WARNING] Negative policy loss: {policy_loss.item():.4f}")
                #    print(f"      ratio: [{ratio.min().item():.3f}, {ratio.max().item():.3f}]")
                #    print(f"      advantages: [{mb_advantages.min().item():.3f}, {mb_advantages.max().item():.3f}]")
                #    print(f"      surr1: [{surr1.min().item():.3f}, {surr1.max().item():.3f}]")
                #    print(f"      surr2: [{surr2.min().item():.3f}, {surr2.max().item():.3f}]")
                
                 # Value loss with clipping (PPO)
                # v_t: current value preds; v_t_old: baseline from buffer
                v_t = value_preds.squeeze(-1) if value_preds.dim() > 1 else value_preds
                v_t_old = mb_old_values.squeeze(-1) if mb_old_values.dim() > 1 else mb_old_values
                v_t_clipped = v_t_old + torch.clamp(v_t - v_t_old,
                                                 -self.config.clip_epsilon,
                                                  self.config.clip_epsilon)
                value_loss_unclipped = (v_t - mb_returns).pow(2)
                value_loss_clipped  = (v_t_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus
                probs = F.softmax(logits.masked_fill(mb_masks == 0, -1e9), dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()
                
                # Teacher loss (DAGGER-lite) - NEW!
                has_teacher = (mb_teacher_idx >= 0)
                if has_teacher.any():
                    # CE loss for teacher labels
                    teacher_log_probs = log_probs_all.gather(1, mb_teacher_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
                    teacher_loss = -teacher_log_probs[has_teacher].mean()
                else:
                    teacher_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = (policy_loss + 
                       self.config.critic_coef * value_loss - 
                       self.current_entropy_coef * entropy +
                       alpha * teacher_loss)  # NEW: add teacher loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_teacher_loss += teacher_loss.item()
                n_updates += 1
        
        # Update stats
        if n_updates > 0:
            self.rollout_stats['policy_loss'].append(total_policy_loss / n_updates)
            self.rollout_stats['value_loss'].append(total_value_loss / n_updates)
            self.rollout_stats['entropy'].append(total_entropy / n_updates)
            self.rollout_stats['teacher_loss'].append(total_teacher_loss / n_updates)
            self.rollout_stats['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        
        # Decay entropy coefficient
        self.current_entropy_coef = max(
            self.config.entropy_min,
            self.current_entropy_coef * self.config.entropy_decay
        )
        
        self.updates += 1
        
        # Print update info periodically
        if self.updates % 10 == 0:
            samples_per_update = self.config.rollout_length * self.config.ppo_epochs
            print(f"\n[PPO Update #{self.updates}]")
            print(f"  Policy loss: {total_policy_loss / n_updates:.4f}")
            print(f"  Value loss: {total_value_loss / n_updates:.4f}")
            print(f"  Entropy: {total_entropy / n_updates:.4f}")
            print(f"  Teacher loss: {total_teacher_loss / n_updates:.4f} (α={alpha:.4f})")
            print(f"  Grad norm: {grad_norm:.4f}")
            print(f"  Entropy coef: {self.current_entropy_coef:.6f}")
            print(f"  Samples/update: {samples_per_update} | Total samples: {self.steps}")
        
        # Clear buffer
        self.buffer.clear()
    
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """Select action with optional teacher signal collection."""
        board_pov = _flip_board(board_copy) if player == -1 else board_copy
        possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
        nA = len(possible_moves)
        
        if nA == 0:
            return []
        
        # Single legal move case
        if nA == 1:
            chosen_move = possible_moves[0]
            chosen_board = possible_boards[0]
            
            terminal_reward = 1.0 if (chosen_board[27] == 15) else 0.0
            shaped_reward = self._compute_shaped_reward(board_pov, chosen_board) if train else 0.0
            total_reward = terminal_reward + shaped_reward
            done = bool(terminal_reward > 0.0)
            
            if train and not self.eval_mode:
                moves_left = 1 + int(dice[0] == dice[1]) - i
                S = self._encode_state(board_pov, moves_left)
                cand_states = np.array([self._encode_state(chosen_board, moves_left - 1)])
                mask = np.ones(1, dtype=np.float32)

                pad_size = self.config.max_actions - 1
                cand_padded = np.pad(cand_states, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')

                # Compute true log-prob and value for the single action
                S_t = torch.as_tensor(S, dtype=torch.float32, device=self.device).unsqueeze(0)
                deltas_t = torch.as_tensor(cand_padded, dtype=torch.float32, device=self.device).unsqueeze(0)
                mask_t = torch.as_tensor(mask_padded, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits_sa, value_sa = self.acnet(S_t, deltas_t, mask_t)
                    log_probs_sa = F.log_softmax(logits_sa.masked_fill(mask_t == 0, -1e9), dim=-1)
                    log_prob_single = log_probs_sa[0, 0].item()
                    value_single = (value_sa.squeeze(-1)[0].item() if value_sa.dim() > 1 else value_sa[0].item())

                self.buffer.push(S, cand_padded, mask_padded, 0, log_prob_single, value_single, total_reward, done, -1)
                self.steps += 1
                if self.buffer.is_ready():
                    self._ppo_update()
            
            if player == -1:
                chosen_move = _flip_move(chosen_move)
            
            return chosen_move
        
        # Multiple legal moves
        moves_left = 1 + int(dice[0] == dice[1]) - i
        S = self._encode_state(board_pov, moves_left)
        
        cand_states = np.stack([
            self._encode_state(board_after, moves_left - 1)
            for board_after in possible_boards
        ], axis=0)
        
        # Cap candidates
        if nA > self.config.max_actions:
            cand_states = cand_states[:self.config.max_actions]
            possible_moves = possible_moves[:self.config.max_actions]
            possible_boards = possible_boards[:self.config.max_actions]
            nA = self.config.max_actions
        
        deltas = cand_states - S
        mask = np.ones(nA, dtype=np.float32)
        
        # Convert to tensors
        S_t = torch.as_tensor(S[None, :], dtype=torch.float32, device=self.device)
        deltas_t = torch.as_tensor(deltas[None, :, :], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            logits, value_t = self.acnet(S_t, deltas_t, mask_t)
            logits = logits.squeeze(0)
            value = value_t.item()
        
        # Action selection
        if train and not self.eval_mode:
            probs = F.softmax(logits, dim=0)
            
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"[WARNING] Invalid probabilities detected. Logits: {logits}")
                probs = torch.ones(nA, device=self.device) / nA
            
            a_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[a_idx] + 1e-9).item()
        else:
            a_idx = torch.argmax(logits).item()
            probs = None
            log_prob = 0.0
        
        chosen_move = possible_moves[a_idx]
        chosen_board = possible_boards[a_idx]
        
        # Reward
        terminal_reward = 1.0 if (chosen_board[27] == 15) else 0.0
        shaped_reward = self._compute_shaped_reward(board_pov, chosen_board) if train else 0.0
        total_reward = terminal_reward + shaped_reward
        done = bool(terminal_reward > 0.0)
        
        # Get teacher label (NEW!)
        teacher_idx = -1
        if train and not self.eval_mode and random.random() < self.config.teacher_sample_rate:
            teacher_idx = self._get_teacher_label(S, deltas, mask)
        
        # Store in buffer
        if train and not self.eval_mode:
            if nA < self.config.max_actions:
                pad_size = self.config.max_actions - nA
                cand_padded = np.pad(cand_states, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')
            else:
                cand_padded = cand_states[:self.config.max_actions]
                mask_padded = mask[:self.config.max_actions]
            
            self.buffer.push(S, cand_padded, mask_padded, a_idx, log_prob, value, total_reward, done, teacher_idx)
            self.steps += 1
            if self.buffer.is_ready():
                self._ppo_update()
        
        if player == -1:
            chosen_move = _flip_move(chosen_move)
        
        return chosen_move
    
    # Compatibility hooks
    def episode_start(self):
        pass
    
    def end_episode(self, outcome, final_board, perspective):
        pass
    
    def game_over_update(self, board, reward):
        pass


# ------------- Module-level interface -------------
_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_ppo.pt")

def _get_agent():
    """Get or create the default agent instance."""
    global _default_agent
    if _default_agent is None:
        device = get_device()
        
        # Try to import pubeval for teacher signal
        try:
            import pubeval_player as pubeval
            print("Pubeval module loaded for teacher signal")
        except ImportError:
            pubeval = None
            print("Warning: Could not import pubeval, teacher signal disabled")
        
        _default_agent = PPOAgent(config=CFG, device=device, pubeval_module=pubeval)
        print(f"PPO agent initialized with improved hyperparameters")
    return _default_agent

def save(path: str = str(CHECKPOINT_PATH)):
    _get_agent().save(path)

def load(path: str = str(CHECKPOINT_PATH), map_location: Union[str, torch.device] = None):
    global _loaded_from_disk
    agent = _get_agent()
    if map_location is None:
        map_location = agent.device
    agent.load(path, map_location)
    _loaded_from_disk = True
    print(f"[Module] Agent loaded from {path}")

def set_eval_mode(is_eval: bool):
    _get_agent().set_eval_mode(is_eval)

def action(board_copy, dice, player, i, train=False, train_config=None):
    global _loaded_from_disk
    if not train and not _loaded_from_disk:
        if CHECKPOINT_PATH.exists():
            try:
                print(f"[Module] Auto-loading checkpoint: {CHECKPOINT_PATH}")
                load(str(CHECKPOINT_PATH), map_location=_get_agent().device)
            except Exception as e:
                print(f"[Module] Could not load checkpoint: {e}")
            _loaded_from_disk = True
    return _get_agent().action(board_copy, dice, player, i, train, train_config)

def episode_start():
    _get_agent().episode_start()

def end_episode(outcome, final_board, perspective):
    _get_agent().end_episode(outcome, final_board, perspective)

def game_over_update(board, reward):
    _get_agent().game_over_update(board, reward)

def __getattr__(name):
    if name in ['_steps', '_updates', '_eval_mode', '_current_entropy_coef', '_rollout_stats', 'CFG']:
        if name == 'CFG':
            return CFG
        agent = _get_agent()
        attr_map = {
            '_steps': 'steps',
            '_updates': 'updates',
            '_eval_mode': 'eval_mode',
            '_current_entropy_coef': 'current_entropy_coef',
            '_rollout_stats': 'rollout_stats',
        }
        return getattr(agent, attr_map.get(name, name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")