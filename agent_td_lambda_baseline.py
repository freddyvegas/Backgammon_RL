#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor-Critic with TD(λ) eligibility traces - Instructor's baseline
Adapted for train.py compatibility

This wraps the instructor's original TD(λ) implementation in a class
that matches the train.py API expectations.
"""

import numpy as np
import torch
from torch.autograd import Variable
from pathlib import Path

# ---- engine imports ----
try:
    import backgammon as Backgammon
except Exception:
    import Backgammon

import flipped_agent

# -------------------- Configuration --------------------
class Config:
    """Configuration for TD(λ) agent."""
    # Feature dimension
    nx = 293  # 24*2*6 + 4 + 1
    H = 146   # Hidden layer (nx // 2)
    
    # Learning rates (instructor's values)
    alpha1 = 0.0005  # layer 1 step size
    alpha2 = 0.0005  # layer 2 step size
    
    # TD(λ) parameters
    gamma = 1.0
    lambda_td = 0.85
    
    # Not used in this agent, but kept for API compatibility
    entropy_coef = 0.0
    entropy_decay = 1.0
    entropy_min = 0.0


class SmallConfig(Config):
    """Small model - same architecture (shallow network)."""
    pass


class MediumConfig(Config):
    """Medium model - same architecture (shallow network)."""
    pass


class LargeConfig(Config):
    """Large model - same architecture (shallow network)."""
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
    
    print(f"\nTD(λ) After-State Baseline Configuration: {size.upper()}")
    print(f"  Architecture: Shallow (nx={cfg.nx}, H={cfg.H})")
    print(f"  Learning rates: α1={cfg.alpha1}, α2={cfg.alpha2}")
    print(f"  TD(λ): γ={cfg.gamma}, λ={cfg.lambda_td}")
    
    return cfg


# -------------------- Features (instructor's encoding) --------------------
def one_hot_encoding(board, nSecondRoll):
    """Instructor's feature encoding."""
    oneHot = np.zeros(293, dtype=np.float32)
    
    # +1 side bins: counts 0,1,2,3,4,5+
    for i in range(0, 5):
        idx = np.where(board[1:25] == i)[0]
        if idx.size > 0:
            oneHot[i*24 + idx] = 1
    idx = np.where(board[1:25] >= 5)[0]
    if idx.size > 0:
        oneHot[5*24 + idx] = 1
    
    # -1 side bins
    for i in range(0, 5):
        idx = np.where(board[1:25] == -i)[0]
        if idx.size > 0:
            oneHot[6*24 + i*24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[6*24 + 5*24 + idx] = 1
    
    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board[25]
    oneHot[12 * 24 + 1] = board[26]
    oneHot[12 * 24 + 2] = board[27]
    oneHot[12 * 24 + 3] = board[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0
    
    return oneHot


# -------------------- Wrapper Class for train.py --------------------
class TDLambdaAgent:
    """
    Wrapper class for instructor's TD(λ) baseline.
    Provides train.py-compatible API.
    """
    
    def __init__(self, config=None, device=None):
        self.config = config or Config()
        self.device = device or torch.device('cpu')
        
        # Initialize parameters (instructor's initialization)
        nx = self.config.nx
        H = self.config.H
        
        self.w1 = Variable(0.1*torch.randn(H, nx, device=self.device, dtype=torch.float32), requires_grad=True)
        self.b1 = Variable(torch.zeros((H, 1), device=self.device, dtype=torch.float32), requires_grad=True)
        self.w2 = Variable(0.1*torch.randn(1, H, device=self.device, dtype=torch.float32), requires_grad=True)
        self.b2 = Variable(torch.zeros((1, 1), device=self.device, dtype=torch.float32), requires_grad=True)
        
        # Eligibility traces
        self.Z_w1 = torch.zeros_like(self.w1.data)
        self.Z_b1 = torch.zeros_like(self.b1.data)
        self.Z_w2 = torch.zeros_like(self.w2.data)
        self.Z_b2 = torch.zeros_like(self.b2.data)
        
        self.Zf_w1 = torch.zeros_like(self.w1.data)
        self.Zf_b1 = torch.zeros_like(self.b1.data)
        self.Zf_w2 = torch.zeros_like(self.w2.data)
        self.Zf_b2 = torch.zeros_like(self.b2.data)
        
        # Episode state
        self.xold = None
        self.xold_flipped = None
        self.I = 1.0
        self.If = 1.0
        self.moveNumber = 0
        
        # Training state (for train.py compatibility)
        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = 0.0  # Not used, but expected by train.py
        
        # Stats tracking (for train.py compatibility)
        self.stats = {
            'td_errors': [],
            'values': [],
            'entropy': [],
            'rewards': [],
        }
        
        print(f"TD(λ) After-State Baseline initialized:")
        print(f"  Device: {self.device}")
        print(f"  Architecture: Shallow (H={H})")
        print(f"  Learning rates: α1={self.config.alpha1}, α2={self.config.alpha2}")
    
    def _greedy_action(self, board, dice, player, nSecondRoll):
        """Greedy action selection for evaluation."""
        flippedplayer = -1
        if flippedplayer == player:
            board_eff = flipped_agent.flip_board(np.copy(board))
            player_eff = -player
        else:
            board_eff = board
            player_eff = player
        
        possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
        na = len(possible_boards)
        if na == 0:
            return []
        
        # Encode after-states
        xa = np.zeros((na, self.config.nx), dtype=np.float32)
        for i in range(na):
            xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
        
        x = Variable(torch.tensor(xa.T, dtype=torch.float32, device=self.device))
        
        # Forward pass (critic)
        h = torch.mm(self.w1, x) + self.b1
        h_tanh = torch.tanh(h)
        y = torch.mm(self.w2, h_tanh) + self.b2
        va = torch.sigmoid(y)
        
        # Select best action
        m = int(torch.argmax(va).item())
        action = possible_moves[m]
        
        if flippedplayer == player:
            action = flipped_agent.flip_move(action)
        
        return action
    
    def _greedy_policy(self, board, dice, player, nRoll):
        """Policy for training (returns action and metadata)."""
        flippedplayer = -1
        nSecondRoll = ((dice[0] == dice[1]) and (nRoll == 0))
        flipped_flag = (flippedplayer == player)
        
        if flipped_flag:
            board_eff = flipped_agent.flip_board(np.copy(board))
            player_eff = -player
        else:
            board_eff = board
            player_eff = player
        
        possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, player_eff)
        na = len(possible_moves)
        if na == 0:
            return [], None, None, None, flipped_flag
        
        # Encode after-states
        xa = np.zeros((na, self.config.nx), dtype=np.float32)
        for i in range(na):
            xa[i, :] = one_hot_encoding(possible_boards[i], nSecondRoll)
        
        x = Variable(torch.tensor(xa.T, dtype=torch.float32, device=self.device))
        
        # Forward pass
        h = torch.mm(self.w1, x) + self.b1
        h_tanh = torch.tanh(h)
        y = torch.mm(self.w2, h_tanh) + self.b2
        va = torch.sigmoid(y)
        
        # Select best action
        m = int(torch.argmax(va).item())
        target = va.data[0, m]
        
        action = possible_moves[m]
        if flipped_flag:
            action = flipped_agent.flip_move(action)
        
        # Features of chosen after-state
        x_selected = Variable(
            torch.tensor(one_hot_encoding(possible_boards[m], nSecondRoll),
                        dtype=torch.float32, device=self.device)
        ).view(self.config.nx, 1)
        
        # Chosen after-state board (in +1 POV)
        chosen_after_eff = possible_boards[m].reshape(-1)
        
        return action, x_selected, target, chosen_after_eff, flipped_flag
    
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """Main action function (train.py API)."""
        nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
        
        # Greedy during eval
        if not train or self.eval_mode:
            return self._greedy_action(np.copy(board_copy), dice, player, nSecondRoll_flag)
        
        # Training: use greedy policy with TD(λ) updates
        act, x, target_val, chosen_after_eff, flipped_flag = self._greedy_policy(
            np.copy(board_copy), dice, player, nRoll=i
        )
        
        if isinstance(act, list) and len(act) == 0:
            return []
        
        # Terminal check
        is_terminal = (chosen_after_eff[27] == 15)
        flippedplayer = -1
        
        # Rewards
        if is_terminal:
            reward = 1.0 if (player != flippedplayer) else 0.0
            rewardf = 1.0 - reward
            tgt = 0.0
        else:
            reward = 0.0
            rewardf = 0.0
            tgt = target_val
        
        # TD(λ) updates (after move 1)
        if (self.moveNumber > 1) and (len(act) > 0):
            # Flipped branch or terminal
            if (flippedplayer == player) or is_terminal:
                if self.xold_flipped is not None:
                    # Forward pass
                    h = torch.mm(self.w1, self.xold_flipped) + self.b1
                    h_tanh = torch.tanh(h)
                    y = torch.mm(self.w2, h_tanh) + self.b2
                    y_sigmoid = torch.sigmoid(y)
                    y_sigmoid.backward()
                    
                    # Update traces
                    self.Zf_w1 = self.config.gamma * self.config.lambda_td * self.Zf_w1 + self.w1.grad.data
                    self.Zf_b1 = self.config.gamma * self.config.lambda_td * self.Zf_b1 + self.b1.grad.data
                    self.Zf_w2 = self.config.gamma * self.config.lambda_td * self.Zf_w2 + self.w2.grad.data
                    self.Zf_b2 = self.config.gamma * self.config.lambda_td * self.Zf_b2 + self.b2.grad.data
                    
                    self.w1.grad.data.zero_()
                    self.b1.grad.data.zero_()
                    self.w2.grad.data.zero_()
                    self.b2.grad.data.zero_()
                    
                    # TD error
                    delta = rewardf + self.config.gamma * tgt - y_sigmoid.item()
                    
                    # Apply updates
                    self.w1.data = self.w1.data + self.config.alpha1 * delta * self.Zf_w1
                    self.b1.data = self.b1.data + self.config.alpha1 * delta * self.Zf_b1
                    self.w2.data = self.w2.data + self.config.alpha2 * delta * self.Zf_w2
                    self.b2.data = self.b2.data + self.config.alpha2 * delta * self.Zf_b2
                    
                    self.If = self.If * self.config.gamma
                    
                    # Track stats
                    self.stats['td_errors'].append(delta)
                    self.stats['values'].append(y_sigmoid.item())
                    self.stats['rewards'].append(rewardf)
                    self.updates += 1
            
            # Non-flipped branch or terminal
            if (flippedplayer != player) or is_terminal:
                if self.xold is not None:
                    # Forward pass
                    h = torch.mm(self.w1, self.xold) + self.b1
                    h_tanh = torch.tanh(h)
                    y = torch.mm(self.w2, h_tanh) + self.b2
                    y_sigmoid = torch.sigmoid(y)
                    y_sigmoid.backward()
                    
                    # Update traces
                    self.Z_w1 = self.config.gamma * self.config.lambda_td * self.Z_w1 + self.w1.grad.data
                    self.Z_b1 = self.config.gamma * self.config.lambda_td * self.Z_b1 + self.b1.grad.data
                    self.Z_w2 = self.config.gamma * self.config.lambda_td * self.Z_w2 + self.w2.grad.data
                    self.Z_b2 = self.config.gamma * self.config.lambda_td * self.Z_b2 + self.b2.grad.data
                    
                    self.w1.grad.data.zero_()
                    self.b1.grad.data.zero_()
                    self.w2.grad.data.zero_()
                    self.b2.grad.data.zero_()
                    
                    # TD error
                    delta = reward + self.config.gamma * tgt - y_sigmoid.item()
                    
                    # Apply updates
                    self.w1.data = self.w1.data + self.config.alpha1 * delta * self.Z_w1
                    self.b1.data = self.b1.data + self.config.alpha1 * delta * self.Z_b1
                    self.w2.data = self.w2.data + self.config.alpha2 * delta * self.Z_w2
                    self.b2.data = self.b2.data + self.config.alpha2 * delta * self.Z_b2
                    
                    self.I = self.I * self.config.gamma
                    
                    # Track stats
                    self.stats['td_errors'].append(delta)
                    self.stats['values'].append(y_sigmoid.item())
                    self.stats['rewards'].append(reward)
                    self.updates += 1
        
        # Cache current side's x
        if len(act) > 0:
            if player == -1:
                self.xold_flipped = x
            else:
                self.xold = x
        
        if not nSecondRoll_flag:
            self.moveNumber += 1
        
        self.steps += 1
        
        return act
    
    def episode_start(self):
        """Reset traces and episode state."""
        self.Z_w1.zero_()
        self.Z_b1.zero_()
        self.Z_w2.zero_()
        self.Z_b2.zero_()
        
        self.Zf_w1.zero_()
        self.Zf_b1.zero_()
        self.Zf_w2.zero_()
        self.Zf_b2.zero_()
        
        self.xold = None
        self.xold_flipped = None
        self.I = 1.0
        self.If = 1.0
        self.moveNumber = 0
    
    def end_episode(self, outcome, final_board, perspective):
        """Episode end hook."""
        pass
    
    def set_eval_mode(self, is_eval: bool):
        """Set evaluation mode."""
        self.eval_mode = is_eval
    
    def save(self, path: str):
        """Save checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'w1': self.w1.data,
            'b1': self.b1.data,
            'w2': self.w2.data,
            'b2': self.b2.data,
            'steps': self.steps,
            'updates': self.updates,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load(self, path: str, map_location=None, load_optimizer: bool = True):
        """Load checkpoint."""
        if map_location is None:
            map_location = self.device
        
        checkpoint = torch.load(path, map_location=map_location)
        
        self.w1.data.copy_(checkpoint['w1'])
        self.b1.data.copy_(checkpoint['b1'])
        self.w2.data.copy_(checkpoint['w2'])
        self.b2.data.copy_(checkpoint['b2'])
        
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        
        print(f"Loaded checkpoint: {path}")
        print(f"  Steps: {self.steps:,}, Updates: {self.updates:,}")


# ============================================================================
# Module-level interface (for train.py compatibility)
# ============================================================================

_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_td_lambda.pt")


def _get_agent():
    global _default_agent
    if _default_agent is None:
        device = torch.device('cpu')  # Instructor's baseline uses CPU
        _default_agent = TDLambdaAgent(config=Config(), device=device)
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
