#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch PPO agent for Backgammon with legal move masking
Class-based design for independent agent instances
Updated with MPS (Apple GPU) support for M1/M2 Macs
"""

from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backgammon

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ---------------- Config ----------------
class Config:
    """Base configuration class."""
    state_dim = 30  # 29 board positions + 1 moves_left counter
    max_actions = 64  # For padding, not actual head size
    
    # PPO hyperparameters
    lr = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    
    # Exploration - NO temperature during training, use entropy bonus instead
    entropy_coef = 0.02
    entropy_decay = 0.9999  # Multiply after each update
    entropy_min = 0.0
    
    critic_coef = 1.0
    eval_temperature = 0.01  # Only for evaluation
    
    # PPO rollout settings
    rollout_length = 512  # Collect this many steps before update
    ppo_epochs = 4  # Number of epochs per rollout
    minibatch_size = 128
    
    # Network architecture (ResMLP) - DEFAULT: LARGE
    model_dim = 512  # Width of ResMLP
    n_blocks = 6  # Number of residual blocks
    
    # Gradient clipping
    grad_clip = 0.5
    max_grad_norm = 0.5
    
    # Weight decay for regularization
    weight_decay = 1e-5
    
    # Reward shaping (KEEP SMALL to avoid swamping terminal reward!)
    use_reward_shaping = True
    pip_reward_scale = 0.01  # Pip count improvement (normalized)
    bear_off_reward = 0.05   # Per checker borne off
    hit_reward = 0.02        # Per hit (currently unused)
    shaping_clip = 0.1       # Clip total shaping per step to ±this value


class SmallConfig(Config):
    """
    Small model for CPU training / quick testing.
    
    ~80K parameters (vs ~1.6M for large)
    Training speed: ~5-10x faster than large
    Performance: Expect ~5-10% lower win rates
    
    Use for:
    - CPU-only training
    - Quick prototyping
    - Testing code changes
    """
    model_dim = 128      # 512 → 128 (4x smaller)
    n_blocks = 3         # 6 → 3 (half the depth)
    rollout_length = 256 # 512 → 256 (faster updates)
    minibatch_size = 64  # 128 → 64 (fits in memory better)
    lr = 2e-4            # Slightly higher LR for faster learning


class MediumConfig(Config):
    """
    Medium model for modest GPUs / balanced training.
    
    ~400K parameters
    Training speed: ~2-3x faster than large
    Performance: Expect ~2-5% lower win rates
    
    Use for:
    - T4 GPU (Google Colab free tier)
    - M1/M2 Mac with MPS
    - Limited GPU time
    - Good balance of speed/performance
    """
    model_dim = 256      # 512 → 256 (2x smaller)
    n_blocks = 4         # 6 → 4 (slightly shallower)
    rollout_length = 384 # 512 → 384 (compromise)
    minibatch_size = 96  # 128 → 96


class LargeConfig(Config):
    """
    Large model (default) for full GPU training.
    
    ~1.6M parameters
    Best performance but slowest training
    
    Use for:
    - A100/V100 GPUs
    - Long training runs
    - Maximum performance
    """
    pass  # Uses defaults from Config


def get_config(size='large'):
    """
    Get configuration for specified model size.
    
    Args:
        size: 'small', 'medium', or 'large'
    
    Returns:
        Config instance
    
    Example:
        cfg = get_config('small')  # For CPU training
        agent = PPOAgent(config=cfg)
    """
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
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Training speed: ~{'5-10x faster' if size == 'small' else '2-3x faster' if size == 'medium' else 'baseline'}")
    
    return cfg


# Default config instance
CFG = Config()


def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS > CPU
    
    Note: All MPS numerical stability issues have been fixed by:
    - Removing LayerNorm
    - Using -1e9 instead of -inf for masking
    - Proper weight initialization
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Set device at module level
CFG.device = get_device()
print(f"PPO agent using device: {CFG.device}")


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

# ------------- Rollout Buffer -------------
class PPORolloutBuffer:
    """Stores on-policy rollouts for PPO updates."""
    def __init__(self, rollout_length=512):
        self.rollout_length = rollout_length
        self.clear()
    
    def clear(self):
        self.states = []
        self.candidate_states = []  # Each is [nA, state_dim]
        self.masks = []  # [nA] - 1 for valid, 0 for padding
        self.actions = []  # Selected action indices
        self.log_probs = []  # Log prob of selected action
        self.values = []
        self.rewards = []
        self.dones = []
        self.size = 0
    
    def push(self, state, cand_states, mask, action_idx, log_prob, value, reward, done):
        self.states.append(state)
        self.candidate_states.append(cand_states)
        self.masks.append(mask)
        self.actions.append(action_idx)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.size += 1
    
    def is_ready(self):
        return self.size >= self.rollout_length
    
    def get(self, max_actions, device):
        """Return all data as torch tensors."""
        # Pad candidate states to max_actions
        padded_cands = []
        padded_masks = []
        
        for cand, mask in zip(self.candidate_states, self.masks):
            if cand.shape[0] < max_actions:
                pad_size = max_actions - cand.shape[0]
                cand_padded = np.pad(cand, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')
            else:
                cand_padded = cand[:max_actions]
                mask_padded = mask[:max_actions]
            
            padded_cands.append(cand_padded)
            padded_masks.append(mask_padded)
        
        # Convert to tensors
        states = torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device)
        cand_states = torch.as_tensor(np.array(padded_cands), dtype=torch.float32, device=device)
        masks = torch.as_tensor(np.array(padded_masks), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array(self.actions), dtype=torch.long, device=device)
        old_log_probs = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        values = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.array(self.dones), dtype=torch.float32, device=device)
        
        return states, cand_states, masks, actions, old_log_probs, values, rewards, dones


# ------------- Actor-Critic Network -------------
class ACNet(nn.Module):
    """
    Delta-based Actor-Critic with ResMLP architecture.
    Given current state s and Δ = s' - s, outputs Q(s, a).
    """
    def __init__(self, state_dim=30, model_dim=512, n_blocks=6):
        super().__init__()
        self.state_dim = state_dim
        self.model_dim = model_dim
        
        # Separate encoders for state and delta
        # Note: Using simpler normalization for MPS stability
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.ReLU()
        )
        
        self.delta_encoder = nn.Sequential(
            nn.Linear(state_dim, model_dim),
            nn.ReLU()
        )
        
        # ResMLP blocks (without LayerNorm for MPS stability)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Linear(model_dim, model_dim),
            )
            for _ in range(n_blocks)
        ])
        
        # Action scoring head
        self.action_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1)
        )
        
        # Value head (operates on state encoding only)
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, 1)
        )
        
        # Initialize weights for numerical stability
        self._init_weights()
    
    def value(self, states):
        return self.value_head(self.state_encoder(states)).squeeze(-1)

    def _init_weights(self):
        """Initialize weights with small values for numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization with small gain for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, states, deltas, mask):
        """
        Args:
            states: [B, state_dim]
            deltas: [B, nA, state_dim]
            mask: [B, nA]
        
        Returns:
            logits: [B, nA]
            values: [B]
        """
        B, nA, _ = deltas.shape
        
        # Encode state once
        state_emb = self.state_encoder(states)  # [B, model_dim]
        
        # Encode all deltas
        deltas_flat = deltas.view(B * nA, -1)  # [B*nA, state_dim]
        delta_emb = self.delta_encoder(deltas_flat)  # [B*nA, model_dim]
        delta_emb = delta_emb.view(B, nA, self.model_dim)  # [B, nA, model_dim]
        
        # Combine: broadcast state embedding to match deltas
        state_emb_expanded = state_emb.unsqueeze(1).expand(B, nA, self.model_dim)  # [B, nA, model_dim]
        combined = state_emb_expanded + delta_emb  # [B, nA, model_dim]
        
        # Apply ResMLP blocks
        combined_flat = combined.view(B * nA, self.model_dim)  # [B*nA, model_dim]
        for block in self.blocks:
            residual = block(combined_flat)
            combined_flat = combined_flat + residual
        combined = combined_flat.view(B, nA, self.model_dim)  # [B, nA, model_dim]
        
        # Score actions
        logits = self.action_head(combined).squeeze(-1)  # [B, nA]
        
        # Apply mask (use large negative instead of -inf to avoid NaN in entropy)
        logits = logits.masked_fill(mask == 0, -1e9)
        
        # Value from state encoding only
        values = self.value_head(state_emb).squeeze(-1)  # [B]
        
        return logits, values
    
    def score_moves_delta(self, states, deltas, mask):
        """Score moves using delta representation (for action selection)."""
        logits, _ = self.forward(states, deltas, mask)
        return logits


# ------------- PPO Agent -------------
class PPOAgent:
    """
    PPO agent with instance-based design for independent training.
    """
    def __init__(self, config=None, device=None):
        self.config = config or CFG
        
        # Device handling: use provided device, otherwise auto-detect
        if device is not None:
            self.device = device
        else:
            self.device = get_device()
        
        print(f"[PPOAgent] Initializing on device: {self.device}")
        
        # Build network
        self.acnet = ACNet(
            state_dim=self.config.state_dim,
            model_dim=self.config.model_dim,
            n_blocks=self.config.n_blocks
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.acnet.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Training state
        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = self.config.entropy_coef
        
        # Rollout buffer
        self.buffer = PPORolloutBuffer(rollout_length=self.config.rollout_length)
        
        # Statistics
        self.rollout_stats = {
            'nA_values': [],
            'advantages': [],
            'masked_entropy': [],
            'value_loss_std': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'grad_norm': [],
        }
        
        print(f"[PPOAgent] Network parameters: {sum(p.numel() for p in self.acnet.parameters()):,}")
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = is_eval
        if is_eval:
            self.acnet.eval()
        else:
            self.acnet.train()
    
    def _encode_state(self, board, moves_left):
        """Encode board state with move counter."""
        state = np.concatenate([board, [moves_left / 4.0]])
        return state.astype(np.float32)
    
    def _compute_shaped_reward(self, old_board, new_board):
        """Compute auxiliary reward shaping (SMALL values only!)."""
        if not self.config.use_reward_shaping:
            return 0.0
        
        reward = 0.0
        
        # Pip count improvement (normalized)
        old_pip = np.sum(np.arange(29) * np.maximum(old_board, 0))
        new_pip = np.sum(np.arange(29) * np.maximum(new_board, 0))
        pip_improvement = (old_pip - new_pip) / 167.0  # 167 = max pip count
        reward += self.config.pip_reward_scale * pip_improvement
        
        # Bear-off progress
        old_borne = old_board[27]
        new_borne = new_board[27]
        if new_borne > old_borne:
            reward += self.config.bear_off_reward * (new_borne - old_borne)
        
        # Clip total shaping
        reward = np.clip(reward, -self.config.shaping_clip, self.config.shaping_clip)
        
        return reward
    
    def save(self, path: str):
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "acnet": self.acnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
            "current_entropy_coef": self.current_entropy_coef,
            "config": {
                "state_dim": self.config.state_dim,
                "max_actions": self.config.max_actions,
                "model_dim": self.config.model_dim,
                "n_blocks": self.config.n_blocks,
                "lr": self.config.lr,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
            }
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str, map_location: Union[str, torch.device] = None):
        """Load agent checkpoint with proper device handling."""
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        self.acnet.load_state_dict(checkpoint["acnet"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = checkpoint.get("steps", 0)
        self.updates = checkpoint.get("updates", 0)
        self.current_entropy_coef = checkpoint.get("current_entropy_coef", self.config.entropy_coef)
        
        print(f"[PPOAgent] Loaded checkpoint from {path}")
        print(f"  Steps: {self.steps}, Updates: {self.updates}")

    def batch_score(self, states_np, cand_states_np, masks_np):
        """
        Batched policy/value for multiple envs.

        Args:
            states_np:     [B, state_dim] float32
            cand_states_np:[B, max_actions, state_dim] float32 (padded)
            masks_np:      [B, max_actions] float32 {0,1}
        Returns:
            logits: [B, max_actions]
            values: [B]
        """
        states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        cand_t   = torch.as_tensor(cand_states_np, dtype=torch.float32, device=self.device)
        masks_t  = torch.as_tensor(masks_np, dtype=torch.float32, device=self.device)

        deltas_t = cand_t - states_t.unsqueeze(1)                 # [B, A, 30]
        with torch.no_grad():
            logits, values = self.acnet(states_t, deltas_t, masks_t)  # logits: [B,A], values: [B]
        return logits, values
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        returns = advantages + values_tensor
        
        return advantages, returns
    
    def _ppo_update(self):
        """Perform PPO update on collected rollout."""
        if not self.buffer.is_ready():
            return
        
        # Get rollout data
        states, cand_states, masks, actions, old_log_probs, values, rewards, dones = \
            self.buffer.get(self.config.max_actions, self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy()
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store statistics
        nA_per_step = masks.sum(dim=1).cpu().numpy()
        self.rollout_stats['nA_values'].extend(nA_per_step.tolist())
        self.rollout_stats['advantages'].extend(advantages.cpu().numpy().tolist())
        
        # PPO epochs
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch training
            indices = torch.randperm(len(states), device=self.device)
            
            for start in range(0, len(states), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                batch_indices = indices[start:end]
                
                # Get batch
                b_states = states[batch_indices]
                b_cand_states = cand_states[batch_indices]
                b_masks = masks[batch_indices]
                b_actions = actions[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]
                
                # Compute deltas
                b_deltas = b_cand_states - b_states.unsqueeze(1)
                
                # Forward pass
                logits, new_values = self.acnet(b_states, b_deltas, b_masks)
                
                # Mask logits with large negative value instead of -inf to avoid NaN
                logits_masked = logits.masked_fill(b_masks == 0, -1e9)
                
                # Compute probabilities
                probs = F.softmax(logits_masked, dim=-1)
                dist_probs = probs.gather(1, b_actions.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(dist_probs + 1e-9)
                
                # PPO clipped loss
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, b_returns)
                
                # Entropy bonus (use finite log to avoid NaN)
                log_probs_finite = torch.log(probs + 1e-12)  # Strictly finite
                entropy = -(probs * log_probs_finite).sum(dim=-1).mean()
                
                # Total loss
                loss = policy_loss + self.config.critic_coef * value_loss - self.current_entropy_coef * entropy
                
                # Check for non-finite loss before backward
                if not torch.isfinite(loss):
                    print(f"[ERROR] Non-finite loss: {loss.item()}, skipping step")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Store for logging
                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN in gradients
                has_nan_grad = False
                for name, param in self.acnet.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient in {name}")
                        has_nan_grad = True
                
                if has_nan_grad:
                    print(f"[ERROR] Skipping optimizer step due to NaN gradients")
                    continue
                
                grad_norm = nn.utils.clip_grad_norm_(self.acnet.parameters(), self.config.max_grad_norm).item()
                
                # Check if grad norm is reasonable
                if not np.isfinite(grad_norm):
                    print(f"[ERROR] Non-finite grad norm: {grad_norm}")
                    continue
                    
                self.optimizer.step()
        
        self.updates += 1
        
        # Decay entropy coefficient
        self.current_entropy_coef = max(self.current_entropy_coef * self.config.entropy_decay, self.config.entropy_min)
        
        # Store stats for this rollout - averages across all epochs/batches
        if epoch_policy_losses:
            self.rollout_stats['policy_loss'].append(np.mean(epoch_policy_losses))
        if epoch_value_losses:
            self.rollout_stats['value_loss'].append(np.mean(epoch_value_losses))
        if epoch_entropies:
            self.rollout_stats['entropy_loss'].append(np.mean(epoch_entropies))
            self.rollout_stats['masked_entropy'].append(np.mean(epoch_entropies))
        
        # Store total loss and grad norm
        if epoch_policy_losses and epoch_value_losses and epoch_entropies:
            total = np.mean(epoch_policy_losses) + self.config.critic_coef * np.mean(epoch_value_losses) - self.current_entropy_coef * np.mean(epoch_entropies)
            self.rollout_stats['total_loss'].append(total)
        
        # Store gradient norm (last one from the update loop)
        if 'grad_norm' in locals():
            self.rollout_stats['grad_norm'].append(grad_norm)
        
        self.rollout_stats['value_loss_std'].append(np.std(epoch_value_losses) if epoch_value_losses else 0)

        
        # Logging every 10 updates
        if self.updates % 10 == 0:
            avg_nA = np.mean(self.rollout_stats['nA_values'][-self.config.rollout_length:]) if self.rollout_stats['nA_values'] else 0
            median_nA = np.median(self.rollout_stats['nA_values'][-self.config.rollout_length:]) if self.rollout_stats['nA_values'] else 0
            avg_advantage = np.mean(np.abs(self.rollout_stats['advantages'][-self.config.rollout_length:])) if self.rollout_stats['advantages'] else 0
            avg_entropy = np.mean(self.rollout_stats['masked_entropy'][-10:]) if self.rollout_stats['masked_entropy'] else 0
            avg_value_std = np.mean(self.rollout_stats['value_loss_std'][-10:]) if self.rollout_stats['value_loss_std'] else 0
            
            # Use epoch averages for clearer signal
            avg_policy = float(np.mean(epoch_policy_losses))
            avg_value = float(np.mean(epoch_value_losses))
            
            samples_per_update = self.config.rollout_length
            
            print(f"\n[Update {self.updates}] Loss: {loss.item():.4f} | "
                  f"Policy(avg): {avg_policy:.4f} | Value(avg): {avg_value:.4f} | "
                  f"Entropy(avg): {avg_entropy:.4f} | Grad: {grad_norm:.4f}")
            print(f"  Entropy coef: {self.current_entropy_coef:.4f} | "
                  f"Mean/Med nA: {avg_nA:.1f}/{median_nA:.1f} | "
                  f"Avg |advantage|: {avg_advantage:.3f} | Value loss std: {avg_value_std:.4f}")
            print(f"  Samples/update: {samples_per_update} | Total samples: {self.steps}")
        
        # Clear buffer for next rollout
        self.buffer.clear()
    
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """Select action for given board state."""
        board_pov = _flip_board(board_copy) if player == -1 else board_copy
        possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
        nA = len(possible_moves)
        
        if nA == 0:
            return []
        
        # Special case: only one legal move
        if nA == 1:
            chosen_move = possible_moves[0]
            chosen_board = possible_boards[0]
            
            # Reward
            terminal_reward = 1.0 if (chosen_board[27] == 15) else 0.0
            shaped_reward = self._compute_shaped_reward(board_pov, chosen_board) if train else 0.0
            total_reward = terminal_reward + shaped_reward
            done = bool(terminal_reward > 0.0)
            
            # Store in rollout buffer if training
            if train and not self.eval_mode:
                moves_left = 1 + int(dice[0] == dice[1]) - i
                S = self._encode_state(board_pov, moves_left)
                cand_states = np.array([self._encode_state(chosen_board, moves_left - 1)])
                mask = np.ones(1, dtype=np.float32)
                
                # Pad to max_actions
                pad_size = self.config.max_actions - 1
                cand_padded = np.pad(cand_states, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')
                
                self.buffer.push(S, cand_padded, mask_padded, 0, 0.0, 0.0, total_reward, done)
                self.steps += 1
                if self.buffer.is_ready():
                    self._ppo_update()
            
            # Flip move back if needed
            if player == -1:
                chosen_move = _flip_move(chosen_move)
            
            return chosen_move
        
        moves_left = 1 + int(dice[0] == dice[1]) - i
        
        # Encode current state
        S = self._encode_state(board_pov, moves_left)
        
        # Encode all candidate resulting states
        cand_states = np.stack([
            self._encode_state(board_after, moves_left - 1)
            for board_after in possible_boards
        ], axis=0)  # [nA, state_dim]
        
        # CRITICAL FIX: Cap candidates before sampling to avoid index out of range
        if nA > self.config.max_actions:
            cand_states = cand_states[:self.config.max_actions]
            possible_moves = possible_moves[:self.config.max_actions]
            possible_boards = possible_boards[:self.config.max_actions]
            nA = self.config.max_actions
        
        # Compute deltas: Δ = s' - s
        deltas = cand_states - S  # [nA, state_dim]
        
        # Create mask
        mask = np.ones(nA, dtype=np.float32)
        
        # Convert to tensors
        S_t = torch.as_tensor(S[None, :], dtype=torch.float32, device=self.device)
        deltas_t = torch.as_tensor(deltas[None, :, :], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            # Use delta-based scoring and get value from forward pass
            logits, value_t = self.acnet(S_t, deltas_t, mask_t)
            logits = logits.squeeze(0)
            value = value_t.item()
        
        # Action selection
        # NO temperature during training - use plain softmax
        # Only use temperature during evaluation
        if train and not self.eval_mode:
            # Training: plain softmax (exploration via entropy bonus)
            probs = F.softmax(logits, dim=0)
            
            # Safety check: if all probs are nan/inf, use uniform distribution
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"[WARNING] Invalid probabilities detected. Logits: {logits}")
                print(f"  nA={nA}, mask={mask}")
                probs = torch.ones(nA, device=self.device) / nA
            
            a_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[a_idx] + 1e-9).item()
        else:
            # Evaluation: strictly greedy (argmax)
            a_idx = torch.argmax(logits).item()
            probs = None
            log_prob = 0.0  # Not used in eval

        chosen_move = possible_moves[a_idx]
        chosen_board = possible_boards[a_idx]

        # Reward
        terminal_reward = 1.0 if (chosen_board[27] == 15) else 0.0
        shaped_reward = self._compute_shaped_reward(board_pov, chosen_board) if train else 0.0
        total_reward = terminal_reward + shaped_reward
        done = bool(terminal_reward > 0.0)

        # Store in rollout buffer
        if train and not self.eval_mode:
            if nA < self.config.max_actions:
                pad_size = self.config.max_actions - nA
                cand_padded = np.pad(cand_states, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')
            else:
                cand_padded = cand_states[:self.config.max_actions]
                mask_padded = mask[:self.config.max_actions]

            self.buffer.push(S, cand_padded, mask_padded, a_idx, log_prob, value, total_reward, done)
            self.steps += 1
            if self.buffer.is_ready():
                self._ppo_update()

        # Flip move back if needed
        if player == -1:
            chosen_move = _flip_move(chosen_move)

        return chosen_move
    
    # Hooks for compatibility
    def episode_start(self):
        pass
    
    def end_episode(self, outcome, final_board, perspective):
        pass
    
    def game_over_update(self, board, reward):
        pass

# ------------- Module-level interface for backward compatibility -------------
# Create a default global instance
_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_ppo.pt")

def _get_agent():
    """Get or create the default agent instance."""
    global _default_agent
    if _default_agent is None:
        device = get_device()
        _default_agent = PPOAgent(config=CFG, device=device)
        print(f"PPO agent initialized")
        print(f"  Device: {device}")
        print(f"  Learning rate: {CFG.lr}")
        print(f"  Rollout length: {CFG.rollout_length}")
        print(f"  PPO epochs: {CFG.ppo_epochs}")
        print(f"  NO temperature during training (using entropy bonus for exploration)")
        print(f"  Eval temperature: {CFG.eval_temperature}")
        print(f"  Entropy coef: {CFG.entropy_coef} -> {CFG.entropy_min}")
        print(f"  Reward shaping: {CFG.use_reward_shaping} (clipped to ±{CFG.shaping_clip})")
    return _default_agent

# Expose agent attributes and methods at module level
def save(path: str = str(CHECKPOINT_PATH)):
    _get_agent().save(path)

def load(path: str = str(CHECKPOINT_PATH), map_location: Union[str, torch.device] = None):
    """Load checkpoint at module level."""
    global _loaded_from_disk
    agent = _get_agent()
    
    # Use agent's device if map_location not specified
    if map_location is None:
        map_location = agent.device
    
    # Load and mark as loaded
    agent.load(path, map_location)
    _loaded_from_disk = True
    
    print(f"[Module] Agent loaded from {path}")

def set_eval_mode(is_eval: bool):
    _get_agent().set_eval_mode(is_eval)

def action(board_copy, dice, player, i, train=False, train_config=None):
    """Module-level action with automatic checkpoint loading."""
    global _loaded_from_disk
    
    # Auto-load checkpoint in eval mode if not already loaded
    if not train and not _loaded_from_disk:
        if CHECKPOINT_PATH.exists():
            try:
                print(f"[Module] Auto-loading checkpoint: {CHECKPOINT_PATH}")
                load(str(CHECKPOINT_PATH), map_location=_get_agent().device)
            except Exception as e:
                print(f"[Module] Could not load checkpoint: {e}")
                print(f"[Module] Continuing with randomly initialized weights")
            _loaded_from_disk = True
    
    return _get_agent().action(board_copy, dice, player, i, train, train_config)

def episode_start():
    _get_agent().episode_start()

def end_episode(outcome, final_board, perspective):
    _get_agent().end_episode(outcome, final_board, perspective)

def game_over_update(board, reward):
    _get_agent().game_over_update(board, reward)

# Expose agent properties (as attributes, not properties)
def __getattr__(name):
    """Forward attribute access to the default agent."""
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
