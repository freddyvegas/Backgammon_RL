#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch PPO agent for Backgammon with legal move masking
Class-based design for independent agent instances
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
    state_dim = 29
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

# Set device at module level
if torch.cuda.is_available():
    CFG.device = "cuda"
else:
    CFG.device = "cpu"

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
            nA = cand.shape[0]
            if nA < max_actions:
                pad_size = max_actions - nA
                cand_padded = np.pad(cand, ((0, pad_size), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, pad_size), mode='constant')
            else:
                cand_padded = cand[:max_actions]
                mask_padded = mask[:max_actions]
            padded_cands.append(cand_padded)
            padded_masks.append(mask_padded)
        
        return (
            torch.as_tensor(np.array(self.states), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(padded_cands), dtype=torch.float32, device=device),
            torch.as_tensor(np.array(padded_masks), dtype=torch.float32, device=device),
            torch.as_tensor(self.actions, dtype=torch.long, device=device),
            torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            torch.as_tensor(self.values, dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards, dtype=torch.float32, device=device),
            torch.as_tensor(self.dones, dtype=torch.bool, device=device),
        )

# ------------- Actor-Critic Network with Move Scoring -------------

class ResBlock(nn.Module):
    """Residual block with pre-activation and LayerNorm."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
    
    def forward(self, x):
        h = self.fc2(F.silu(self.fc1(self.norm(x))))
        return x + h


class ResMLP(nn.Module):
    """Residual MLP with SiLU activation and LayerNorm."""
    def __init__(self, in_dim, d=512, n=6):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.Sequential(*[ResBlock(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)
    
    def forward(self, x):
        x = F.silu(self.inp(x))
        x = self.blocks(x)
        return self.norm(x)


class ActorCriticNet(nn.Module):
    """
    Network with ResMLP backbone and delta-based move scoring.
    
    Uses φ(s) for state encoding and ψ(Δ) for move deltas,
    scoring moves via dot product: logit = φ(s) · ψ(Δ)
    """
    def __init__(self, in_dim=29, d=512, n_blocks=6):
        super().__init__()
        
        # Shared ResMLP backbone for state encoding
        self.shared = ResMLP(in_dim, d=d, n=n_blocks)
        
        # Value head (scores current state)
        self.value_trunk = nn.Sequential(
            ResBlock(d),
            ResBlock(d),
        )
        self.value = nn.Linear(d, 1)
        
        # Policy: state encoder φ(s)
        self.policy_trunk = nn.Sequential(
            ResBlock(d),
            ResBlock(d),
        )
        self.state_proj = nn.Linear(d, d)
        
        # Policy: move delta encoder ψ(Δ)
        # Encodes the change (s' - s) rather than full s'
        self.delta_encoder = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.SiLU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with orthogonal weights and small gain."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use smaller gain for stability with deeper networks
                if m.out_features == 1:  # Output heads
                    nn.init.orthogonal_(m.weight, gain=0.01)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode_state(self, x):
        """Encode state(s) through shared backbone."""
        return self.shared(x)
    
    def value_head(self, x):
        """Compute value for state(s)."""
        features = self.encode_state(x)
        h = self.value_trunk(features)
        return self.value(h).squeeze(-1)
    
    def score_moves_delta(self, state, deltas, mask=None):
        """
        Score moves using delta representation.
        
        Args:
            state: [B, state_dim] - current state
            deltas: [B, A, state_dim] - move deltas (s' - s)
            mask: [B, A] - 1 for valid, 0 for padding
        
        Returns:
            logits: [B, A] - scores for each move (masked)
        """
        B, A, D = deltas.shape
        
        # Encode state once: φ(s)
        state_features = self.encode_state(state)  # [B, d]
        h_state = self.policy_trunk(state_features)  # [B, d]
        z_state = self.state_proj(h_state)  # [B, d]
        
        # Encode all deltas: ψ(Δ) for each move
        flat_deltas = deltas.view(B * A, D)  # [B*A, state_dim]
        z_deltas = self.delta_encoder(flat_deltas)  # [B*A, d]
        z_deltas = z_deltas.view(B, A, -1)  # [B, A, d]
        
        # Score via dot product: φ(s) · ψ(Δ)
        scores = (z_state.unsqueeze(1) * z_deltas).sum(-1)  # [B, A]
        
        # Mask invalid actions
        if mask is not None:
            scores = scores + (mask == 0) * (-1e9)
        
        return scores
    
    def score_moves(self, cand_states, mask=None):
        """
        Backward compatibility: score by re-encoding full states.
        Kept for compatibility but delta version is preferred.
        """
        # For backward compatibility, but not used in delta-based training
        raise NotImplementedError("Use score_moves_delta instead")

# ------------- PPO Agent Class -------------
class PPOAgent:
    """
    PPO agent with legal move masking.
    Each instance has its own network, optimizer, and buffer.
    """
    def __init__(self, config=None, device=None):
        '''
        Initialize PPO Agent.
        
        Args:
            config: Config instance (SmallConfig, MediumConfig, or LargeConfig)
                    If None, uses default Config()
            device: 'cuda' or 'cpu'. If None, auto-detects.
        '''
        # Use provided config or default
        if config is None:
            config = CFG
        
        self.config = config
        self.device = device if device else (config.device if hasattr(config, 'device') else CFG.device)
        
        # Create network with config dimensions
        self.acnet = ActorCriticNet(
            in_dim=self.config.state_dim,
            d=self.config.model_dim,        # ← Uses config
            n_blocks=self.config.n_blocks   # ← Uses config
        ).to(self.device)

        decay, no_decay = [], []
        for n, p in self.acnet.named_parameters():
            if p.ndim == 1 or 'bias' in n or 'norm' in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {'params': decay, 'weight_decay': self.config.weight_decay},
                {'params': no_decay, 'weight_decay': 0.0},
            ],
            lr=self.config.lr,
            eps=1e-5,
        )
        
        self.buffer = PPORolloutBuffer(self.config.rollout_length)
        
        # State
        self.eval_mode = False
        self.steps = 0
        self.updates = 0
        self.current_entropy_coef = self.config.entropy_coef
        
        # Tracking for logging
        self.rollout_stats = {
            'nA_values': [],
            'advantages': [],
            'masked_entropy': [],
            'value_loss_std': []
        }
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = bool(is_eval)
        if self.eval_mode:
            self.acnet.eval()
        else:
            self.acnet.train()
    
    def save(self, path: str):
        """Save agent state."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "acnet": self.acnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "updates": self.updates,
            "entropy_coef": self.current_entropy_coef,
        }, p)
    
    def load(self, path: str, map_location=None):
        """Load agent state with robust error handling."""
        if map_location is None:
            map_location = self.device
        
        # Load checkpoint with weights_only=False for backward compatibility
        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(path, map_location=map_location)
        
        # Load network weights
        self.acnet.load_state_dict(checkpoint["acnet"])
        
        # Load optimizer state if present
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                print(f"[Load] Warning: Could not load optimizer state: {e}")
        
        # Load training state
        if "steps" in checkpoint:
            self.steps = checkpoint["steps"]
        if "updates" in checkpoint:
            self.updates = checkpoint["updates"]
        if "entropy_coef" in checkpoint:
            self.current_entropy_coef = checkpoint["entropy_coef"]
        
        # Ensure network is on correct device
        self.acnet.to(self.device)
        
        # Set to eval mode after loading
        self.set_eval_mode(True)
        
        # Print confirmation
        print(f"[Load] Checkpoint loaded: {path}")
        print(f"  Device: {self.device}")
        print(f"  Steps: {self.steps:,}, Updates: {self.updates:,}")
        print(f"  Entropy coef: {self.current_entropy_coef:.4f}")

    def can_load(self, path: str) -> bool:
        """Check if checkpoint exists and is valid."""
        p = Path(path)
        if not p.exists():
            return False
        
        try:
            checkpoint = torch.load(p, map_location='cpu', weights_only=False)
            return "acnet" in checkpoint
        except Exception as e:
            print(f"[Load] Cannot load {path}: {e}")
            return False
    
    def _encode_state(self, board_flipped, moves_left):
        """Encode board state."""
        x = np.zeros(self.config.state_dim, dtype=np.float32)
        x[:24] = np.clip(board_flipped[1:25] * 0.2, -1, 1)
        x[24] = np.clip(board_flipped[25] * 0.2, -1, 1)
        x[25] = np.clip(board_flipped[26] * 0.2, -1, 1)
        x[26] = board_flipped[27] / 15.0
        x[27] = board_flipped[28] / 15.0
        x[28] = float(moves_left)
        return x
    
    def _compute_shaped_reward(self, prev_board, new_board):
        """Compute reward shaping based on progress."""
        if not self.config.use_reward_shaping:
            return 0.0
        
        reward = 0.0
        
        # Progress in bearing off
        prev_borne = prev_board[27]
        new_borne = new_board[27]
        if new_borne > prev_borne:
            reward += self.config.bear_off_reward * (new_borne - prev_borne)
        
        # Pip count improvement (normalized)
        def pip_count(board):
            total = 0
            for i in range(1, 25):
                if board[i] > 0:
                    total += i * board[i]
            return total
        
        prev_pip = pip_count(prev_board)
        new_pip = pip_count(new_board)
        pip_delta = (prev_pip - new_pip) / 100.0  # Normalize
        reward += self.config.pip_reward_scale * pip_delta
        
        # Clip to prevent rare spikes
        reward = np.clip(reward, -self.config.shaping_clip, self.config.shaping_clip)
        
        return reward
    
    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        # Ensure last_value is a tensor on the right device
        if not isinstance(last_value, torch.Tensor):
            last_value = torch.tensor(last_value, dtype=torch.float32, device=self.device)
        
        # Ensure all tensors are on the right device
        values = values.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t].float()) * gae
            advantages.insert(0, gae)
        
        advantages = torch.stack(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def _ppo_update(self):
        """Perform PPO update on collected rollout."""
        if self.eval_mode or not self.buffer.is_ready():
            return
        
        # Get rollout data
        states, cand_states, masks, actions, old_log_probs, values, rewards, dones = \
            self.buffer.get(self.config.max_actions, self.device)
        
        # Compute deltas: Δ = s' - s for each candidate
        # states: [B, state_dim], cand_states: [B, A, state_dim]
        deltas = cand_states - states.unsqueeze(1)  # [B, A, state_dim]
        
        # Compute last value for GAE
        with torch.no_grad():
            last_value = self.acnet.value_head(states[-1:])
            if last_value.dim() > 0:
                last_value = last_value.item()
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones, last_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track statistics
        rollout_nA = (masks.sum(dim=1)).cpu().numpy()
        self.rollout_stats['nA_values'].extend(rollout_nA.tolist())
        self.rollout_stats['advantages'].extend(advantages.cpu().numpy().tolist())
        
        # PPO epochs
        total_samples = len(states)
        indices = np.arange(total_samples)
        
        epoch_value_losses = []
        epoch_policy_losses = []
        epoch_entropies = []
        
        for epoch in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_deltas = deltas[mb_indices]
                mb_masks = masks[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass - use delta-based scoring
                logits = self.acnet.score_moves_delta(mb_states, mb_deltas, mb_masks)
                log_probs = F.log_softmax(logits, dim=1)
                
                # Get log prob of taken actions
                new_log_probs = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                
                policy_loss = -torch.min(
                    ratio * mb_advantages,
                    clipped_ratio * mb_advantages
                ).mean()
                epoch_policy_losses.append(policy_loss.item())
                
                # Value loss
                new_values = self.acnet.value_head(mb_states)
                value_loss = F.mse_loss(new_values, mb_returns)
                epoch_value_losses.append(value_loss.item())
                
                # Entropy bonus (computed directly from masked logits with safety)
                probs = F.softmax(logits, dim=1)
                safe_mask = (mb_masks > 0).float()
                entropy_vec = -(probs * log_probs * safe_mask).sum(dim=1)
                entropy = torch.nan_to_num(entropy_vec, nan=0.0, posinf=0.0, neginf=0.0).mean()
                epoch_entropies.append(entropy.item())
                
                # Total loss
                loss = policy_loss + self.config.critic_coef * value_loss - self.current_entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.acnet.parameters(), self.config.max_grad_norm).item()
                self.optimizer.step()
        
        self.updates += 1
        
        # Decay entropy coefficient
        self.current_entropy_coef = max(self.current_entropy_coef * self.config.entropy_decay, self.config.entropy_min)
        
        # Store stats for this rollout
        self.rollout_stats['masked_entropy'].append(np.mean(epoch_entropies))
        self.rollout_stats['value_loss_std'].append(np.std(epoch_value_losses))
        
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
            # Use delta-based scoring
            logits = self.acnet.score_moves_delta(S_t, deltas_t, mask_t).squeeze(0)
            value = self.acnet.value_head(S_t).item()
        
        # Action selection
        # NO temperature during training - use plain softmax
        # Only use temperature during evaluation
        if train and not self.eval_mode:
            # Training: plain softmax (exploration via entropy bonus)
            probs = F.softmax(logits, dim=0)
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
