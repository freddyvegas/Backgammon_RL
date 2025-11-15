#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro-move Actor-Critic agent for Backgammon (A2C-Micro)

Implementation of Part B from the assignment PDF:
- Fixed 156-action space (26 sources × 6 dice)
- Legality masking per micro-step
- State-value critic (no after-states)
- TD(λ) with eligibility traces
- Micro-rollout within each turn
"""

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from utils import _flip_board, _flip_move, get_device
import backgammon

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for micro-move A2C agent."""
    state_dim = 293  # one-hot encoding dimension
    n_micro_actions = 156  # 26 sources × 6 dice
    
    # Learning rates
    lr_critic = 1e-4
    lr_actor = 1e-4
    
    # TD(λ) parameters
    gamma = 1.0  # discount factor (as specified in PDF)
    lambda_td = 0.9  # eligibility trace decay
    
    # Entropy regularization
    entropy_coef = 0.01
    entropy_decay = 0.9999
    entropy_min = 0.001
    
    # Network architecture (ResMLP)
    model_dim = 512
    n_blocks = 6
    
    # Gradient clipping
    max_grad_norm = 0.5
    
    # Evaluation
    eval_temperature = 0.01
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class SmallConfig(Config):
    """Small model for CPU training."""
    model_dim = 128
    n_blocks = 3


class MediumConfig(Config):
    """Medium model for modest GPUs."""
    model_dim = 256
    n_blocks = 4


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
    
    print(f"\nMicro-A2C Configuration: {size.upper()}")
    print(f"  Architecture: dim={cfg.model_dim}, blocks={cfg.n_blocks}")
    print(f"  Learning rates: critic={cfg.lr_critic}, actor={cfg.lr_actor}")
    print(f"  TD(λ): γ={cfg.gamma}, λ={cfg.lambda_td}")
    print(f"  Micro-actions: {cfg.n_micro_actions}")
    
    return cfg


CFG = Config()
CFG.device = get_device()
print(f"Micro-A2C agent using device: {CFG.device}")

# ============================================================================
# Micro-action utilities
# ============================================================================
def one_hot_encoding(board29, nSecondRoll: bool):
    """
    board29: float/int array length 29 in +1 POV
    nSecondRoll: True on the first move of doubles, else False
    returns: np.float32 shape (293,)
    """
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1, dtype=np.float32)

    # +1 side bins: counts 0,1,2,3,4,5+
    for i in range(0, 5):
        idx = np.where(board29[1:25] == i)[0]         # <-- no -1
        if idx.size > 0:
            oneHot[i * 24 + idx] = 1.0
    idx = np.where(board29[1:25] >= 5)[0]
    if idx.size > 0:
        oneHot[5 * 24 + idx] = 1.0

    # -1 side bins: counts 0,1,2,3,4,5+ (negative)
    for i in range(0, 5):
        idx = np.where(board29[1:25] == -i)[0]        # <-- no -1
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1.0
    idx = np.where(board29[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[6 * 24 + 5 * 24 + idx] = 1.0

    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board29[25]
    oneHot[12 * 24 + 1] = board29[26]
    oneHot[12 * 24 + 2] = board29[27]
    oneHot[12 * 24 + 3] = board29[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0

    return oneHot



def encode_micro_action(src: int, die: int) -> int:
    """
    Encode (source, die) as action index.
    Formula: index = 6 * src + (die - 1)
    
    Args:
        src: Source point (0-25, where 0 is no-op/bar indicator)
        die: Die value (1-6)
    
    Returns:
        Action index in [0, 155]
    """
    assert 0 <= src <= 25, f"Invalid source: {src}"
    assert 1 <= die <= 6, f"Invalid die: {die}"
    return 6 * src + (die - 1)


def decode_micro_action(action_idx: int) -> Tuple[int, int]:
    """
    Decode action index to (source, die).
    
    Args:
        action_idx: Action index in [0, 155]
    
    Returns:
        (src, die) tuple
    """
    src = action_idx // 6
    die = (action_idx % 6) + 1
    return src, die


def build_legality_mask(board29: np.ndarray, die_value: int, player: int = 1) -> np.ndarray:
    """
    Build 156-dimensional legality mask for a single die.
    
    Args:
        board29: Board state (length 29) from player's POV
        die_value: Die value (1-6)
        player: Player (+1 or -1), but should always be +1 for POV
    
    Returns:
        mask: np.ndarray of shape (156,) with 1.0 for legal, 0.0 for illegal
    """
    mask = np.zeros(156, dtype=np.float32)
    
    # Get legal single-die moves from backgammon.py
    legal_moves = backgammon.legal_move(board29, die_value, player)
    
    if len(legal_moves) == 0:
        # Only no-op is legal (src=0)
        mask[encode_micro_action(0, die_value)] = 1.0
        return mask
    
    # Mark each legal move
    for move in legal_moves:
        src = int(move[0])
        action_idx = encode_micro_action(src, die_value)
        mask[action_idx] = 1.0
    
    return mask


# ============================================================================
# Network Architecture
# ============================================================================

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


class MicroActorCritic(nn.Module):
    """
    Actor-Critic network for micro-actions.
    
    Architecture:
    - Shared trunk: state encoder + residual blocks
    - Actor head: 156 logits (fixed micro-action space)
    - Critic head: single value output
    """
    
    def __init__(self, state_dim=293, model_dim=512, n_blocks=6):
        super().__init__()
        self.state_dim = state_dim
        self.model_dim = model_dim
        
        # State encoder
        self.state_enc = nn.Linear(state_dim, model_dim)
        
        # Residual MLP blocks (shared trunk)
        self.blocks = nn.ModuleList([
            ResMLPBlock(model_dim) for _ in range(n_blocks)
        ])
        
        # Actor head: 156 logits
        self.actor_head = nn.Linear(model_dim, 156)
        
        # Critic head: single value
        self.value_head = nn.Linear(model_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.state_enc.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, state_features, mask):
        """
        Forward pass.
        
        Args:
            state_features: (B, state_dim) - one-hot encoded states
            mask: (B, 156) - legality mask
        
        Returns:
            logits: (B, 156) - masked logits
            values: (B,) - state values
        """
        # Encode state
        x = F.relu(self.state_enc(state_features))
        
        # Process through residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Get logits and values
        logits = self.actor_head(x)  # (B, 156)
        values = torch.sigmoid(self.value_head(x)).squeeze(-1)  # bound to [0,1]
        
        # Apply mask (use -1e9 instead of -inf for MPS compatibility)
        logits = logits.masked_fill(mask == 0, -1e9)
        
        return logits, values
    
    def get_value(self, state_features):
        """Get value estimate only (for critic-only forward pass)."""
        x = F.relu(self.state_enc(state_features))
        for block in self.blocks:
            x = block(x)
        values = torch.sigmoid(self.value_head(x)).squeeze(-1)  # bound to [0,1]
        return values


# ============================================================================
# Eligibility Traces
# ============================================================================

class TDLambdaTraces:
    """
    Eligibility traces for TD(λ) learning.
    
    Maintains separate traces for actor and critic parameters.
    """
    
    def __init__(self, acnet, lambda_val=0.9):
        self.lambda_val = lambda_val
        self.acnet = acnet
        self.critic_traces = {}
        self.actor_traces = {}
        self._initialize_traces()
    
    def _initialize_traces(self):
        """Initialize trace tensors (zeros)."""
        for name, param in self.acnet.named_parameters():
            self.critic_traces[name] = torch.zeros_like(param.data)
            self.actor_traces[name] = torch.zeros_like(param.data)
    
    def reset(self):
        """Reset all traces to zero (call at episode start)."""
        for name in self.critic_traces:
            self.critic_traces[name].zero_()
        for name in self.actor_traces:
            self.actor_traces[name].zero_()
    
    def update_critic(self, td_error: float, alpha_critic: float):
        """
        Update critic traces and apply TD error.
        
        Formula: e_V = λ * e_V + ∇V(s)
                 φ ← φ + α_c * δ * e_V
        
        Args:
            td_error: TD error δ
            alpha_critic: Critic learning rate
        """
        for name, param in self.acnet.named_parameters():
            if 'value' in name and param.grad is not None:
                # Update trace: e = λ * e + ∇V
                self.critic_traces[name].mul_(self.lambda_val).add_(param.grad.data)
                
                # Apply: φ ← φ + α * δ * e
                param.data.add_(self.critic_traces[name], alpha=alpha_critic * td_error)
    
    def update_actor(self, td_error: float, alpha_actor: float):
        """
        Update actor traces and apply TD error.
        
        Formula: e_π = λ * e_π + ∇log π(a|s)
                 θ ← θ + α_a * δ * e_π
        
        Args:
            td_error: TD error δ
            alpha_actor: Actor learning rate
        """
        for name, param in self.acnet.named_parameters():
            if 'actor' in name and param.grad is not None:
                # Update trace: e = λ * e + ∇log π
                self.actor_traces[name].mul_(self.lambda_val).add_(param.grad.data)
                
                # Apply: θ ← θ + α * δ * e
                param.data.add_(self.actor_traces[name], alpha=alpha_actor * td_error)


# ============================================================================
# Micro-move A2C Agent
# ============================================================================

class MicroA2CAgent:
    """
    Micro-move Actor-Critic agent for Backgammon.
    
    Implements the specification from Part B of the assignment PDF.
    """
    
    def __init__(self, config=None, device=None):
        self.config = config or CFG
        self.device = device or get_device()
        
        # Network
        self.acnet = MicroActorCritic(
            state_dim=self.config.state_dim,
            model_dim=self.config.model_dim,
            n_blocks=self.config.n_blocks
        ).to(self.device)
        
        # Eligibility traces
        self.traces = TDLambdaTraces(self.acnet, lambda_val=self.config.lambda_td)
        
        # Training state
        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = self.config.entropy_coef
        
        # Stats tracking
        self.stats = {
            'td_errors': [],
            'values': [],
            'entropy': [],
            'rewards': [],
        }
        
        print(f"MicroA2CAgent initialized:")
        print(f"  Device: {self.device}")
        print(f"  Learning rates: critic={self.config.lr_critic}, actor={self.config.lr_actor}")
        print(f"  TD(λ): γ={self.config.gamma}, λ={self.config.lambda_td}")
        print(f"  Parameters: {sum(p.numel() for p in self.acnet.parameters()):,}")
    
    def _encode_state(self, board29, is_second_roll: bool) -> np.ndarray:
        """
        Encode board state as one-hot features.
        
        Args:
            board29: Board state (length 29)
            is_second_roll: True only if doubles AND first micro-step of turn
        
        Returns:
            features: np.ndarray of shape (293,)
        """
        return one_hot_encoding(board29.astype(np.float32), False)
    
    def select_micro_action(self, state_features, mask, training=True):
        """
        Select a micro-action given state and legality mask.
        
        Args:
            state_features: np.ndarray of shape (293,)
            mask: np.ndarray of shape (156,) - legality mask
            training: If True, sample stochastically; else greedy
        
        Returns:
            action_idx: Selected action index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        # Convert to tensors
        state_t = torch.tensor(state_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.acnet(state_t, mask_t)
            logits = logits.squeeze(0)  # (156,)
            value = value.item()
        
        # Action selection
        if training and not self.eval_mode:
            # Stochastic sampling
            probs = F.softmax(logits, dim=0)
            
            # Safety check
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"[WARNING] Invalid probabilities. Using uniform over legal actions.")
                probs = torch.tensor(mask, device=self.device)
                probs = probs / probs.sum()
            
            action_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action_idx] + 1e-9).item()
        else:
            # Greedy selection
            action_idx = torch.argmax(logits).item()
            log_prob = 0.0
        
        return action_idx, log_prob, value
    
    def _apply_micro_action(self, board29, action_idx, die_value):
        """
        Apply a single micro-action to the board.
        
        Args:
            board29: Current board state
            action_idx: Micro-action index
            die_value: Die value being used
        
        Returns:
            next_board: Board after applying action
            is_noop: True if action was no-op
        """
        src, _ = decode_micro_action(action_idx)
        
        if src == 0:
            # No-op: no move possible
            return board29.copy(), True
        
        # Get legal moves for this die to find destination
        legal_moves = backgammon.legal_move(board29, die_value, player=1)
        
        # Find the move matching this source
        move = None
        for m in legal_moves:
            if int(m[0]) == src:
                move = m
                break
        
        if move is None:
            # Should not happen if mask is correct
            print(f"[WARNING] No legal move found for src={src}, die={die_value}")
            return board29.copy(), True
        
        # Apply the move
        next_board = backgammon.update_board(board29, move.reshape(1, 2), player=1)
        return next_board, False
    
    def micro_rollout_turn(self, board29, dice, training=True):
        """
        Execute a complete turn as a sequence of micro-steps.
        
        Args:
            board29: Initial board state from +1 POV
            dice: Dice roll (length 2)
            training: If True, collect trajectory for learning
        
        Returns:
            trajectory: List of micro-step transitions
            final_board: Board state after turn
            full_move: Complete move sequence (for compatibility)
        """
        # Each call is responsible for at most two micro-steps
        dice_remaining = [dice[0], dice[1]]
        
        # Compute is_doubles once for this turn (Bug 2 fix)
        is_doubles = (dice[0] == dice[1])
 
        trajectory = []
        current_board = board29.copy()
        move_sequence = []  # Track moves for reconstruction
        micro_step = 0  # Track which micro-step we're on
        
        while dice_remaining:
            die = dice_remaining[0]
            
            # Build legality mask
            mask = build_legality_mask(current_board, die, player=1)
            
            # Check if only no-op is legal (Bug 3 already fixed)
            only = np.flatnonzero(mask == 1.0)
            if len(only) == 1:
                src, _ = decode_micro_action(int(only[0]))
                if src == 0:
                    # This die is effectively dead, drop it and try the next one
                    dice_remaining.pop(0)
                    micro_step += 1
                    continue
           
            # Encode state with correct second_roll flag (Bug 2 fix)
            # True only for doubles AND first micro-step
            is_second_roll = is_doubles and (micro_step == 0)
            state_features = self._encode_state(current_board, False)
            
            # Select action
            action_idx, log_prob, value = self.select_micro_action(
                state_features, mask, training=training
            )
            
            # Apply action
            next_board, is_noop = self._apply_micro_action(current_board, action_idx, die)
            
            # Record move for reconstruction (if not no-op)
            if not is_noop:
                src, _ = decode_micro_action(action_idx)
                # Find actual destination from legal moves
                legal = backgammon.legal_move(current_board, die, player=1)
                for m in legal:
                    if int(m[0]) == src:
                        # Store as (2,) array
                        move_sequence.append(np.array([m[0], m[1]], dtype=np.int32))
                        break
            
            # Remove die from remaining
            dice_remaining.pop(0)
            micro_step += 1
            
            # Compute reward - check terminal state every micro-step (Bug 4 fix)
            done = backgammon.game_over(next_board)
            reward = 0.0
            if done:
                # +1 for win, -1 for loss (from +1 POV)
                reward = 1.0 if next_board[27] == 15 else -1.0
            
            # Store transition
            if training:
                trajectory.append({
                    'state': state_features,
                    'mask': mask,
                    'action': action_idx,
                    'log_prob': log_prob,
                    'value': value,
                    'reward': reward,
                    'done': done,
                    'board': current_board.copy(),
                })
            
            current_board = next_board
            
            # Break if game is over
            if done:
                break
        
        # Reconstruct full move - ensure compatible format with backgammon.py
        if len(move_sequence) == 0:
            full_move = np.empty((0, 2), dtype=np.int32)
        elif len(move_sequence) == 1:
            full_move = move_sequence[0].reshape(1, 2)
        else:
            # At most 2 moves per call
            full_move = np.vstack(move_sequence[:2]).reshape(2, 2)
 
        return trajectory, current_board, full_move
    
    def td_lambda_update(self, trajectory):
        """
        Perform TD(λ) updates for a micro-rollout trajectory.
        
        Args:
            trajectory: List of micro-step transitions
        """
        if len(trajectory) == 0:
            return
        
        # Process each micro-step
        for t, transition in enumerate(trajectory):
            state = transition['state']
            action = transition['action']
            mask = transition['mask']
            reward = transition['reward']
            done = transition['done']
            value = transition['value']
            
            # Get next value
            if done or t == len(trajectory) - 1:
                next_value = 0.0
            else:
                next_state = trajectory[t + 1]['state']
                next_mask = trajectory[t + 1]['mask']
                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                next_mask_t = torch.tensor(next_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    _, next_value_t = self.acnet(next_state_t, next_mask_t)
                    next_value = next_value_t.item()
            
            # TD error: δ = r + γV(s') - V(s)
            td_error = reward + (0.0 if done else self.config.gamma * next_value) - value
            
            # Convert to tensors
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Forward pass (with gradients)
            self.acnet.zero_grad()
            logits, value_pred = self.acnet(state_t, mask_t)
            
            # Critic gradient: ∇V(s)
            value_pred.backward(retain_graph=True)
            
            # Apply critic update with traces
            self.traces.update_critic(td_error, self.config.lr_critic)
            
            # Actor gradient: ∇log π(a|s)
            self.acnet.zero_grad()
            logits, _ = self.acnet(state_t, mask_t)
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, action]
            
            # Add entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum()
            
            # Actor loss (negative for gradient ascent)
            actor_loss = log_prob + self.current_entropy_coef * entropy
            actor_loss.backward()
            
            # Apply actor update with traces
            self.traces.update_actor(td_error, self.config.lr_actor)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), self.config.max_grad_norm)
            
            # Track stats
            self.stats['td_errors'].append(td_error)
            self.stats['values'].append(value)
            self.stats['entropy'].append(entropy.item())
            self.stats['rewards'].append(reward)
            
            self.steps += 1
        
        self.updates += 1
        
        # Decay entropy coefficient
        self.current_entropy_coef = max(
            self.config.entropy_min,
            self.current_entropy_coef * self.config.entropy_decay
        )
        
        # Print stats periodically
        if self.updates % 100 == 0:
            recent_td = np.mean(self.stats['td_errors'][-100:])
            recent_v = np.mean(self.stats['values'][-100:])
            recent_ent = np.mean(self.stats['entropy'][-100:])
            print(f"\n[Micro-A2C Update #{self.updates}]")
            print(f"  Avg TD error: {recent_td:.4f}")
            print(f"  Avg value: {recent_v:.4f}")
            print(f"  Avg entropy: {recent_ent:.4f}")
            print(f"  Entropy coef: {self.current_entropy_coef:.6f}")
            print(f"  Total steps: {self.steps:,}")
    
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """
        Select action for a turn (compatible with training loop API).
        
        Args:
            board_copy: Board state
            dice: Dice roll
            player: Player (+1 or -1)
            i: Move index within turn (not used in micro-approach)
            train: If True, perform learning updates
            train_config: Training configuration (not used)
        
        Returns:
            move: Move array compatible with backgammon.py
        """
        # Flip to +1 POV
        board_pov = _flip_board(board_copy) if player == -1 else board_copy.copy()
        
        # Check for legal moves using standard backgammon.legal_moves
        possible_moves, _ = backgammon.legal_moves(board_pov, dice, player=1)
        if len(possible_moves) == 0:
            # No legal moves - return empty move
            return np.array([], dtype=np.int32).reshape(0, 2)
        
        # Execute micro-rollout
        trajectory, final_board, full_move = self.micro_rollout_turn(
            board_pov, dice, training=train
        )
        
        # TD(λ) update if training
        if train and not self.eval_mode:
            self.td_lambda_update(trajectory)
        
        # Ensure move has correct shape
        if full_move.size == 0:
            # Empty move case
            full_move = np.array([], dtype=np.int32).reshape(0, 2)
        elif full_move.ndim == 1:
            # Single move stored as 1D - reshape to (1, 2)
            if len(full_move) == 2:
                full_move = full_move.reshape(1, 2)
            else:
                # Malformed move, return empty
                full_move = np.array([], dtype=np.int32).reshape(0, 2)
        # else: full_move already has shape (k, 2)
        
        # Flip move back if needed
        if player == -1 and full_move.size > 0:
            full_move = _flip_move(full_move)
        
        return full_move
    
    def episode_start(self):
        """Called at episode start - reset eligibility traces."""
        self.traces.reset()
    
    def end_episode(self, outcome, final_board, perspective):
        """Called at episode end."""
        pass
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = is_eval
        if is_eval:
            self.acnet.eval()
        else:
            self.acnet.train()
    
    def save(self, path: str):
        """Save agent checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'acnet': self.acnet.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'entropy_coef': self.current_entropy_coef,
            'config': {
                'state_dim': self.config.state_dim,
                'model_dim': self.config.model_dim,
                'n_blocks': self.config.n_blocks,
                'lr_critic': self.config.lr_critic,
                'lr_actor': self.config.lr_actor,
                'lambda_td': self.config.lambda_td,
            }
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load(self, path: str, map_location: Union[str, torch.device] = None, load_optimizer: bool = True):
        """Load agent checkpoint."""
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Check if architecture matches
        saved_config = checkpoint.get('config', {})
        if saved_config:
            if (saved_config.get('state_dim', self.config.state_dim) != self.config.state_dim or
                saved_config.get('model_dim', self.config.model_dim) != self.config.model_dim or
                saved_config.get('n_blocks', self.config.n_blocks) != self.config.n_blocks):
                
                print(f"  Rebuilding network with saved architecture")
                self.acnet = MicroActorCritic(
                    state_dim=saved_config.get('state_dim', self.config.state_dim),
                    model_dim=saved_config.get('model_dim', self.config.model_dim),
                    n_blocks=saved_config.get('n_blocks', self.config.n_blocks)
                ).to(self.device)
                
                # Reinitialize traces
                self.traces = TDLambdaTraces(self.acnet, lambda_val=self.config.lambda_td)
        
        self.acnet.load_state_dict(checkpoint['acnet'])
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        self.current_entropy_coef = checkpoint.get('entropy_coef', self.config.entropy_coef)
        
        print(f"Loaded checkpoint: {path}")
        print(f"  Steps: {self.steps:,}")
        print(f"  Updates: {self.updates:,}")


# ============================================================================
# Module-level interface (for compatibility)
# ============================================================================

_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_micro_a2c.pt")


def _get_agent():
    """Get or create the default agent instance."""
    global _default_agent
    if _default_agent is None:
        device = get_device()
        _default_agent = MicroA2CAgent(config=CFG, device=device)
        print(f"Micro-A2C agent initialized")
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


def __getattr__(name):
    if name in ['steps', 'updates', 'eval_mode', 'current_entropy_coef', 'stats', 'CFG']:
        if name == 'CFG':
            return CFG
        agent = _get_agent()
        return getattr(agent, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == "__main__":
    # Quick test
    print("\n" + "="*70)
    print("MICRO-A2C AGENT TEST")
    print("="*70)
    
    # Create agent
    agent = MicroA2CAgent(config=get_config('small'), device='cpu')
    
    # Test action encoding/decoding
    print("\nTesting action encoding:")
    for src in [0, 1, 6, 24, 25]:
        for die in [1, 6]:
            idx = encode_micro_action(src, die)
            src_dec, die_dec = decode_micro_action(idx)
            print(f"  src={src:2d}, die={die} → idx={idx:3d} → src={src_dec:2d}, die={die_dec}")
            assert src == src_dec and die == die_dec
    
    # Test legality mask
    print("\nTesting legality mask:")
    board = backgammon.init_board()
    dice_roll = backgammon.roll_dice()
    print(f"  Dice: {dice_roll}")
    for die in dice_roll:
        mask = build_legality_mask(board, die, player=1)
        n_legal = int(mask.sum())
        print(f"  Die {die}: {n_legal} legal actions")
    
    # Test action selection
    print("\nTesting action selection:")
    trajectory, final_board, move = agent.micro_rollout_turn(board, dice_roll, training=False)
    print(f"  Trajectory length: {len(trajectory)}")
    print(f"  Final move: {move}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

