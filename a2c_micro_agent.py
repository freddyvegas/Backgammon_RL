#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro-move Actor-Critic agent for Backgammon (Simplified TD(λ) version)

Following the instructor's style:
- Shallow network with raw tensor parameters (w1, b1, w2_actor, b2_actor, w2_critic, b2_critic)
- Manual parameter updates with eligibility traces
- Fixed 156-action space with legality masking
- State-value critic (no after-states)
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import random

from utils import _flip_board, _flip_move, get_device
import backgammon

# Set seeds
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for micro-move A2C agent."""
    # Feature dimension (one-hot encoding)
    nx = 293  # 24*2*6 + 4 + 1
    
    # Micro-action space
    n_micro_actions = 156  # 26 sources × 6 dice
    
    # Network architecture
    H = 146  # Hidden layer size (nx // 2, like instructor's)
    
    # Learning rates
    lr_critic = 5e-6   # Very gentle for manual updates
    lr_actor = 5e-6
    
    # TD(λ) parameters
    gamma = 1.0
    lambda_td = 0.7
    
    # Entropy regularization
    entropy_coef = 0.02
    entropy_decay = 0.99995
    entropy_min = 0.005
    
    # Gradient clipping
    max_grad_norm = 0.2

CFG = Config()

class Config:
    """Configuration for micro-move A2C agent."""
    nx = 293
    n_micro_actions = 156
    H = 146
    
    # MUCH GENTLER learning rates
    lr_critic = 1e-6   # 5x smaller
    lr_actor = 1e-6    # 5x smaller
    
    # TD(λ) parameters
    gamma = 1.0
    lambda_td = 0.8    # Higher for better credit assignment
    
    # Entropy - slower decay
    entropy_coef = 0.05
    entropy_decay = 0.999995  # Much slower
    entropy_min = 0.01
    
    # More aggressive clipping
    max_grad_norm = 0.1

class SmallConfig(Config):
    """Small model for CPU training."""
    H = 64
    lr_critic = 5e-7   # Even gentler for small model
    lr_actor = 5e-7
    lambda_td = 0.75
    max_grad_norm = 0.08

class MediumConfig(Config):
    """Medium model for modest GPUs."""
    H = 96
    lr_critic = 4e-6
    lr_actor = 4e-6
    lambda_td = 0.65


class LargeConfig(Config):
    """Large model (default)."""
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
    print(f"  Hidden dim: {cfg.H}")
    print(f"  Learning rates: critic={cfg.lr_critic}, actor={cfg.lr_actor}")
    print(f"  TD(λ): γ={cfg.gamma}, λ={cfg.lambda_td}")
    print(f"  Micro-actions: {cfg.n_micro_actions}")
    
    return cfg

# Set device
device = get_device()
print(f"Micro-A2C (simplified) using device: {device}")

# ============================================================================
# Feature encoding (identical to instructor's)
# ============================================================================

def one_hot_encoding(board, nSecondRoll):
    """
    Encode board state as one-hot features.
    board: array of length 29 from +1 POV
    nSecondRoll: True if doubles AND first move of turn
    """
    oneHot = np.zeros(24 * 2 * 6 + 4 + 1, dtype=np.float32)
    
    # +1 side bins: counts 0,1,2,3,4,5+
    for i in range(0, 5):
        idx = np.where(board[1:25] == i)[0]
        if idx.size > 0:
            oneHot[i * 24 + idx] = 1.0
    idx = np.where(board[1:25] >= 5)[0]
    if idx.size > 0:
        oneHot[5 * 24 + idx] = 1.0
    
    # -1 side bins: counts 0,1,2,3,4,5+ (negative)
    for i in range(0, 5):
        idx = np.where(board[1:25] == -i)[0]
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1.0
    idx = np.where(board[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[6 * 24 + 5 * 24 + idx] = 1.0
    
    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0
    
    return oneHot

# ============================================================================
# Micro-action utilities
# ============================================================================

def encode_micro_action(src: int, die: int) -> int:
    """Encode (source, die) as action index: 6 * src + (die - 1)"""
    assert 0 <= src <= 25
    assert 1 <= die <= 6
    return 6 * src + (die - 1)

def decode_micro_action(action_idx: int):
    """Decode action index to (src, die)"""
    src = action_idx // 6
    die = (action_idx % 6) + 1
    return src, die

def build_legality_mask_multi(board29, dice_remaining, player=1):
    """
    Build 156-dimensional legality mask for remaining dice.
    Returns mask with 1.0 for legal actions, 0.0 for illegal.
    """
    mask = np.zeros(156, dtype=np.float32)
    
    any_legal = False
    for die in dice_remaining:
        legal_moves = backgammon.legal_move(board29, die, player)
        if len(legal_moves) == 0:
            continue
        any_legal = True
        for move in legal_moves:
            src = int(move[0])
            action_idx = encode_micro_action(src, die)
            mask[action_idx] = 1.0
    
    if not any_legal:
        # No legal moves: only no-ops for each distinct die
        for die in set(dice_remaining):
            mask[encode_micro_action(0, die)] = 1.0
    
    return mask

# ============================================================================
# Network Parameters (Raw Tensors)
# ============================================================================

class MicroA2CNetwork:
    """
    Shallow network with raw tensor parameters (instructor's style).
    
    Architecture:
      h = tanh(w1 @ x + b1)        # Hidden layer (H,)
      logits = w2_actor @ h + b2_actor  # Actor head (156,)
      value = sigmoid(w2_critic @ h + b2_critic)  # Critic head (1,)
    """
    
    def __init__(self, nx=293, H=146, n_actions=156, device='cpu'):
        self.nx = nx
        self.H = H
        self.n_actions = n_actions
        self.device = device
        
        # Shared trunk
        self.w1 = torch.randn(H, nx, device=device, dtype=torch.float32) * 0.05
        self.b1 = torch.zeros(H, 1, device=device, dtype=torch.float32)
        
        # Actor head (156 logits)
        self.w2_actor = torch.randn(n_actions, H, device=device, dtype=torch.float32) * 0.01
        self.b2_actor = torch.zeros(n_actions, 1, device=device, dtype=torch.float32)
        
        # Critic head (1 value)
        self.w2_critic = torch.randn(1, H, device=device, dtype=torch.float32) * 0.1
        self.b2_critic = torch.zeros(1, 1, device=device, dtype=torch.float32)
        
        # Mark all as leaf tensors (no autograd by default)
        for param in [self.w1, self.b1, self.w2_actor, self.b2_actor, 
                      self.w2_critic, self.b2_critic]:
            param.requires_grad = False
    
    def forward(self, x, mask):
        """
        Forward pass.
        
        Args:
            x: (nx, N) - features for N states
            mask: (N, 156) - legality mask
        
        Returns:
            logits: (N, 156) - masked logits
            values: (N,) - value estimates
        """
        N = x.shape[1]  # Number of states in batch
        
        # Shared hidden layer
        h = torch.tanh(torch.mm(self.w1, x) + self.b1)  # (H, N)
        
        # Actor logits: (n_actions, H) @ (H, N) = (n_actions, N) -> transpose -> (N, n_actions)
        logits = torch.mm(self.w2_actor, h).T  # (N, 156)
        # Broadcast bias: (156, 1) -> (156,)
        logits = logits + self.b2_actor.squeeze()  # (N, 156)
        
        # Critic value: (1, H) @ (H, N) = (1, N) -> squeeze -> (N,)
        values = torch.mm(self.w2_critic, h).squeeze(0)  # (N,)
        # Add bias scalar
        values = values + self.b2_critic.squeeze()  # (N,)
        
        # Apply mask
        logits = logits.masked_fill(mask == 0, -1e9)
        
        return logits, values
    
    def parameters(self):
        """Return list of all parameters."""
        return [self.w1, self.b1, self.w2_actor, self.b2_actor, 
                self.w2_critic, self.b2_critic]

# ============================================================================
# Eligibility Traces
# ============================================================================

class EligibilityTraces:
    """TD(λ) eligibility traces for manual parameter updates."""
    
    def __init__(self, network, lambda_val=0.7):
        self.network = network
        self.lambda_val = lambda_val
        
        # Initialize traces for each parameter
        self.critic_traces = [torch.zeros_like(p) for p in network.parameters()]
        self.actor_traces = [torch.zeros_like(p) for p in network.parameters()]
    
    def reset(self):
        """Reset all traces to zero."""
        for trace in self.critic_traces:
            trace.zero_()
        for trace in self.actor_traces:
            trace.zero_()
    
    def decay(self):
        """Decay all traces by λ."""
        for trace in self.critic_traces:
            trace.mul_(self.lambda_val)
        for trace in self.actor_traces:
            trace.mul_(self.lambda_val)

# ============================================================================
# Micro-move A2C Agent
# ============================================================================

class MicroA2CAgent:
    """Simplified Micro-A2C agent with manual TD(λ) updates."""
    
    def __init__(self, config=None, device=None):
        self.config = config or CFG
        self.device = device or get_device()
        
        # Network
        self.net = MicroA2CNetwork(
            nx=self.config.nx,
            H=self.config.H,
            n_actions=self.config.n_micro_actions,
            device=self.device
        )
        
        # Eligibility traces
        self.traces = EligibilityTraces(self.net, lambda_val=self.config.lambda_td)
        
        # Training state
        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = self.config.entropy_coef
        
        # Stats
        self.stats = {
            'td_errors': [],
            'values': [],
            'entropy': [],
            'rewards': [],
        }
        
        print(f"MicroA2CAgent (simplified) initialized:")
        print(f"  Device: {self.device}")
        print(f"  Hidden dim: {self.config.H}")
        print(f"  Learning rates: {self.config.lr_critic}, {self.config.lr_actor}")
    
    def _encode_state(self, board29, is_second_roll: bool):
        """Encode board state as one-hot features."""
        return one_hot_encoding(board29.astype(np.float32), is_second_roll)
    
    @torch.no_grad()
    def select_micro_action(self, state_features, mask, training=True):
        """Select a micro-action given state and legality mask."""
        # Convert to tensors
        x = torch.tensor(state_features, dtype=torch.float32, device=self.device).view(self.config.nx, 1)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward pass
        logits, value = self.net.forward(x, mask_t)
        logits = logits.squeeze(0)  # (156,)
        value = value.item()
        
        # Action selection
        if training and not self.eval_mode:
            # Stochastic sampling
            probs = F.softmax(logits, dim=0)
            
            # Safety check
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.tensor(mask, device=self.device)
                probs = probs / probs.sum()
            
            action_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[action_idx] + 1e-9).item()
        else:
            # Greedy selection
            action_idx = torch.argmax(logits).item()
            log_prob = 0.0
        
        return action_idx, log_prob, value
    
    def _apply_micro_action(self, board29, action_idx):
        """Apply a single micro-action to the board."""
        src, die_value = decode_micro_action(action_idx)
        
        if src == 0:
            # No-op
            return board29.copy(), True
        
        # Get legal moves for this die
        legal_moves = backgammon.legal_move(board29, die_value, player=1)
        
        move = None
        for m in legal_moves:
            if int(m[0]) == src:
                move = m
                break
        
        if move is None:
            return board29.copy(), True
        
        # Apply the move
        next_board = backgammon.update_board(board29, move.reshape(1, 2), player=1)
        return next_board, False
    
    def micro_rollout_turn(self, board29, dice, training=True):
        """Execute a complete turn as a sequence of micro-steps."""
        # Doubles: four dice; otherwise: two dice
        if dice[0] == dice[1]:
            dice_remaining = [dice[0]] * 4
            is_doubles = True
        else:
            dice_remaining = [dice[0], dice[1]]
            is_doubles = False
        
        trajectory = []
        current_board = board29.copy()
        move_sequence = []
        micro_step = 0
        
        while dice_remaining:
            # Build legality mask
            mask = build_legality_mask_multi(current_board, dice_remaining, player=1)
            
            # Encode state (second-roll flag only for doubles AND first step)
            is_second_roll = is_doubles and (micro_step == 0)
            state_features = self._encode_state(current_board, is_second_roll)
            
            # Select action
            action_idx, log_prob, value = self.select_micro_action(
                state_features, mask, training=training
            )
            
            # Decode action
            src, die_value = decode_micro_action(action_idx)
            
            # Apply micro-action
            next_board, is_noop = self._apply_micro_action(current_board, action_idx)
            
            # Record move if not no-op
            if not is_noop:
                legal = backgammon.legal_move(current_board, die_value, player=1)
                for m in legal:
                    if int(m[0]) == src:
                        move_sequence.append(np.array([m[0], m[1]], dtype=np.int32))
                        break
            
            # Remove die
            try:
                die_idx = dice_remaining.index(die_value)
                dice_remaining.pop(die_idx)
            except ValueError:
                dice_remaining.pop(0)
            
            micro_step += 1
            
            # Check if game is over
            done = backgammon.game_over(next_board)
            reward = 0.0
            if done:
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
                })
            
            current_board = next_board
            
            if done:
                break
        
        # Reconstruct full move
        if len(move_sequence) == 0:
            full_move = np.empty((0, 2), dtype=np.int32)
        elif len(move_sequence) == 1:
            full_move = move_sequence[0].reshape(1, 2)
        else:
            full_move = np.vstack(move_sequence[:2]).reshape(-1, 2)
        
        return trajectory, current_board, full_move
    
    def td_lambda_update(self, trajectory):
        """
        Perform TD(λ) updates with manual parameter updates.
        
        Follows instructor's style:
        1. Compute gradients manually
        2. Update eligibility traces
        3. Apply parameter updates: param += lr * δ * trace
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
            
            # Convert to tensors
            x = torch.tensor(state, dtype=torch.float32, device=self.device).view(self.config.nx, 1)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Enable gradients temporarily
            for param in self.net.parameters():
                param.requires_grad = True
            
            # ============================================================
            # STEP 1: Compute V(sₜ) and its gradient
            # ============================================================
            _, value_t = self.net.forward(x, mask_t)
            value_current = value_t.item()
            
            # Compute ∇V for critic trace
            value_t.backward()
            
            # Store critic gradients
            critic_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                           for p in self.net.parameters()]
            
            # Clear gradients
            for param in self.net.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # ============================================================
            # STEP 2: Compute V(sₜ₊₁) for TD error
            # ============================================================
            if done or t == len(trajectory) - 1:
                next_value = 0.0
            else:
                next_state = trajectory[t + 1]['state']
                next_mask = trajectory[t + 1]['mask']
                next_x = torch.tensor(next_state, dtype=torch.float32, device=self.device).view(self.config.nx, 1)
                next_mask_t = torch.tensor(next_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    for param in self.net.parameters():
                        param.requires_grad = False
                    _, next_value_t = self.net.forward(next_x, next_mask_t)
                    next_value = next_value_t.item()
            
            # TD error
            td_error = reward + (0.0 if done else self.config.gamma * next_value) - value_current
            
            # CRITICAL: Clip TD error to prevent divergence
            td_error = np.clip(td_error, -1.0, 1.0)
            
            # ============================================================
            # STEP 3: Update critic traces and parameters
            # ============================================================
            # Only update: w1, b1 (shared), w2_critic, b2_critic
            param_list = [self.net.w1, self.net.b1, self.net.w2_critic, self.net.b2_critic]
            grad_indices = [0, 1, 4, 5]  # Indices in parameters() list
            
            for param_idx, param in zip(grad_indices, param_list):
                grad = critic_grads[param_idx]
                
                # Update trace: e = λ * e + ∇V
                self.traces.critic_traces[param_idx].mul_(self.config.lambda_td).add_(grad)
                
                # Clip trace
                trace_norm = self.traces.critic_traces[param_idx].norm().item()
                if trace_norm > self.config.max_grad_norm:
                    self.traces.critic_traces[param_idx].mul_(self.config.max_grad_norm / trace_norm)
                
                # Apply update: param += α * δ * trace
                param.data.add_(self.traces.critic_traces[param_idx], alpha=self.config.lr_critic * td_error)
            
            # ============================================================
            # STEP 4: Compute ∇log π(aₜ|sₜ) for actor trace
            # ============================================================
            for param in self.net.parameters():
                param.requires_grad = True
            
            logits, _ = self.net.forward(x, mask_t)
            log_probs = F.log_softmax(logits.masked_fill(mask_t == 0, -1e9), dim=-1)
            log_prob = log_probs[0, action]
            
            # Entropy bonus
            probs = F.softmax(logits.masked_fill(mask_t == 0, -1e9), dim=-1)
            entropy = -(probs * log_probs).sum()
            
            # Actor objective
            actor_obj = log_prob + self.current_entropy_coef * entropy
            actor_obj.backward()
            
            # Store actor gradients
            actor_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                          for p in self.net.parameters()]
            
            # Clear gradients
            for param in self.net.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # ============================================================
            # STEP 5: Update actor traces and parameters
            # ============================================================
            for i, (param, grad) in enumerate(zip(self.net.parameters(), actor_grads)):
                # Update trace: e = λ * e + ∇log π
                self.traces.actor_traces[i].mul_(self.config.lambda_td).add_(grad)
                
                # Clip trace
                trace_norm = self.traces.actor_traces[i].norm().item()
                if trace_norm > self.config.max_grad_norm:
                    self.traces.actor_traces[i].mul_(self.config.max_grad_norm / trace_norm)
                
                # Apply update: param += α * δ * trace
                param.data.add_(self.traces.actor_traces[i], alpha=self.config.lr_actor * td_error)
            
            # Disable gradients after update
            for param in self.net.parameters():
                param.requires_grad = False
            
            # Track stats
            self.stats['td_errors'].append(td_error)
            self.stats['values'].append(value_current)
            self.stats['entropy'].append(entropy.item())
            self.stats['rewards'].append(reward)
            
            self.steps += 1
        
        self.updates += 1
        
        # Decay entropy
        self.current_entropy_coef = max(
            self.config.entropy_min,
            self.current_entropy_coef * self.config.entropy_decay
        )
        
        # Print stats
        if self.updates % 100 == 0:
            recent_td = np.mean(self.stats['td_errors'][-100:]) if self.stats['td_errors'] else 0.0
            recent_v = np.mean(self.stats['values'][-100:]) if self.stats['values'] else 0.0
            recent_ent = np.mean(self.stats['entropy'][-100:]) if self.stats['entropy'] else 0.0
            recent_rewards = np.mean(self.stats['rewards'][-100:]) if self.stats['rewards'] else 0.0
            
            # Check trace norms
            critic_trace_norm = sum(t.norm().item() for t in self.traces.critic_traces) / len(self.traces.critic_traces)
            actor_trace_norm = sum(t.norm().item() for t in self.traces.actor_traces) / len(self.traces.actor_traces)
            
            print(f"\n[Micro-A2C Update #{self.updates}]")
            print(f"  Avg TD error: {recent_td:.4f}")
            print(f"  Avg value: {recent_v:.4f}")
            print(f"  Avg entropy: {recent_ent:.4f}")
            print(f"  Avg reward: {recent_rewards:.4f}")
            print(f"  Critic trace norm: {critic_trace_norm:.4f}")
            print(f"  Actor trace norm: {actor_trace_norm:.4f}")
            print(f"  Total steps: {self.steps:,}")
    
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """Select action for a turn (API compatibility)."""
        # Flip to +1 POV
        board_pov = _flip_board(board_copy) if player == -1 else board_copy.copy()
        
        # Check for legal moves
        possible_moves, _ = backgammon.legal_moves(board_pov, dice, player=1)
        if len(possible_moves) == 0:
            return np.array([], dtype=np.int32).reshape(0, 2)
        
        # Execute micro-rollout
        trajectory, final_board, full_move = self.micro_rollout_turn(
            board_pov, dice, training=train
        )
        
        # TD(λ) update if training
        if train and not self.eval_mode:
            self.td_lambda_update(trajectory)
        
        # Ensure correct shape
        if full_move.size == 0:
            full_move = np.array([], dtype=np.int32).reshape(0, 2)
        elif full_move.ndim == 1 and len(full_move) == 2:
            full_move = full_move.reshape(1, 2)
        
        # Flip move back if needed
        if player == -1 and full_move.size > 0:
            full_move = _flip_move(full_move)
        
        return full_move
    
    def episode_start(self):
        """Called at episode start."""
        self.traces.reset()
    
    def end_episode(self, outcome, final_board, perspective):
        """Called at episode end."""
        pass
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = is_eval
    
    def save(self, path: str):
        """Save agent checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'w1': self.net.w1,
            'b1': self.net.b1,
            'w2_actor': self.net.w2_actor,
            'b2_actor': self.net.b2_actor,
            'w2_critic': self.net.w2_critic,
            'b2_critic': self.net.b2_critic,
            'steps': self.steps,
            'updates': self.updates,
            'entropy_coef': self.current_entropy_coef,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load(self, path: str, map_location=None, load_optimizer: bool = True):
        """Load agent checkpoint."""
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        
        self.net.w1.copy_(checkpoint['w1'])
        self.net.b1.copy_(checkpoint['b1'])
        self.net.w2_actor.copy_(checkpoint['w2_actor'])
        self.net.b2_actor.copy_(checkpoint['b2_actor'])
        self.net.w2_critic.copy_(checkpoint['w2_critic'])
        self.net.b2_critic.copy_(checkpoint['b2_critic'])
        
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        self.current_entropy_coef = checkpoint.get('entropy_coef', self.config.entropy_coef)
        
        print(f"Loaded checkpoint: {path}")
        print(f"  Steps: {self.steps:,}, Updates: {self.updates:,}")


# ============================================================================
# Module-level interface
# ============================================================================

_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_micro_a2c.pt")


def _get_agent():
    global _default_agent
    if _default_agent is None:
        _default_agent = MicroA2CAgent(config=CFG, device=device)
    return _default_agent


def save(path: str = str(CHECKPOINT_PATH)):
    _get_agent().save(path)


def load(path: str = str(CHECKPOINT_PATH), map_location=None):
    global _loaded_from_disk
    agent = _get_agent()
    if map_location is None:
        map_location = agent.device
    agent.load(path, map_location)
    _loaded_from_disk = True


def set_eval_mode(is_eval: bool):
    _get_agent().set_eval_mode(is_eval)


def action(board_copy, dice, player, i, train=False, train_config=None):
    global _loaded_from_disk
    if not train and not _loaded_from_disk:
        if CHECKPOINT_PATH.exists():
            try:
                load(str(CHECKPOINT_PATH), map_location=_get_agent().device)
            except Exception as e:
                print(f"Could not load checkpoint: {e}")
            _loaded_from_disk = True
    return _get_agent().action(board_copy, dice, player, i, train, train_config)


def episode_start():
    _get_agent().episode_start()


def end_episode(outcome, final_board, perspective):
    _get_agent().end_episode(outcome, final_board, perspective)
