#!/usr/bin/env python3
"""Self-contained PPO convolutional agent for tournament.py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import backgammon


_DEFAULT_CHECKPOINT = os.environ.get(
    "PPO_TOURNAMENT_CHECKPOINT",
    "latest_ppo_cnn_large_20251208_213011.pt",
)


def _detect_device(requested: str | None = None) -> torch.device:
    """Return a torch.device, defaulting to the best available accelerator."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32,
)


def _flip_board(board: np.ndarray) -> np.ndarray:
    out = np.empty(29, dtype=board.dtype)
    out[:] = -board[_FLIP_IDX]
    return out


def _flip_move(move: Sequence[Sequence[int]]) -> List[List[int]]:
    if move is None:
        return []
    if hasattr(move, "__len__") and len(move) == 0:
        return []
    flipped = []
    for step in move:
        src, dst = step
        flipped.append([int(_FLIP_IDX[src]), int(_FLIP_IDX[dst])])
    return flipped


def encode_board_cnn_torch(board29: torch.Tensor, n_second_roll: torch.Tensor | float | int):
    """Encode raw 29-dim boards into convolutional channels + globals."""
    if board29.dim() == 1:
        board29 = board29.unsqueeze(0)
    board29 = board29.to(dtype=torch.float32)
    batch = board29.size(0)

    points = board29[:, 1:25]
    ours = torch.clamp(points, min=0)
    opp = torch.clamp(-points, min=0)

    pos_bins = [
        (ours == 1).float(),
        (ours == 2).float(),
        ((ours >= 3) & (ours <= 5)).float(),
        (ours >= 6).float(),
    ]
    neg_bins = [
        (opp == 1).float(),
        (opp == 2).float(),
        ((opp >= 3) & (opp <= 5)).float(),
        (opp >= 6).float(),
    ]

    idx = torch.linspace(0.0, 1.0, 24, device=board29.device, dtype=torch.float32)
    idx = idx.view(1, 24).expand(batch, -1)
    home = torch.zeros_like(idx)
    home[:, :6] = 1.0

    X_points = torch.stack(pos_bins + neg_bins + [idx, home], dim=1)

    bar_p = board29[:, 25]
    bar_o = board29[:, 26]
    off_p = board29[:, 27] / 15.0
    off_o = board29[:, 28].abs() / 15.0

    if not torch.is_tensor(n_second_roll):
        n_second_roll = torch.tensor(n_second_roll, device=board29.device, dtype=torch.float32)
    else:
        n_second_roll = n_second_roll.to(device=board29.device, dtype=torch.float32)
    if n_second_roll.dim() == 0:
        n_second_roll = n_second_roll.expand(batch)
    n_second_roll = n_second_roll.reshape(batch)

    X_global = torch.stack(
        [bar_p, bar_o, off_p, off_o, n_second_roll],
        dim=-1,
    )

    return X_points, X_global


@dataclass
class ConvConfig:
    raw_board_dim: int = 29
    second_roll_index: int = 29
    state_dim: int = 30  # raw board + second-roll flag
    max_actions: int = 64
    use_raw_board_inputs: bool = True

    conv_in_channels: int = 10
    conv_channels: int = 96
    conv_layers: int = 4
    conv_kernel_size: int = 3
    conv_pool: str = "mean"

    model_dim: int = 640
    n_blocks: int = 8
    ff_mult: float = 2.0
    resid_dropout: float = 0.1
    delta_hidden_mult: float = 1.5
    value_hidden_mult: float = 0.5


class SmallConvConfig(ConvConfig):
    model_dim = 256
    n_blocks = 4
    conv_channels = 48
    conv_layers = 2
    resid_dropout = 0.05


class MediumConvConfig(ConvConfig):
    model_dim = 384
    n_blocks = 5
    conv_channels = 72
    conv_layers = 3
    resid_dropout = 0.05


def get_config(size: str = "large") -> ConvConfig:
    size = size.lower()
    mapping = {
        "small": SmallConvConfig,
        "medium": MediumConvConfig,
        "large": ConvConfig,
    }
    if size not in mapping:
        raise ValueError(f"Unknown size '{size}' (expected one of {list(mapping)})")
    return mapping[size]()


class ResMLPBlock(nn.Module):
    """Simple residual MLP block used inside the actor-critic trunk."""

    def __init__(self, dim: int, ff_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim, int(dim * ff_mult))
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x + self.ff(self.ln(x))


class PPOConvActorCritic(nn.Module):
    """Actor-critic with CNN encoder for state and candidate boards."""

    def __init__(self, cfg: ConvConfig):
        super().__init__()
        self.cfg = cfg
        self.board_dim = cfg.raw_board_dim
        self.flag_index = cfg.second_roll_index

        channels = cfg.conv_channels
        conv_layers = []
        in_ch = cfg.conv_in_channels
        for _ in range(cfg.conv_layers):
            conv_layers.append(
                nn.Conv1d(in_ch, channels, kernel_size=cfg.conv_kernel_size, padding=cfg.conv_kernel_size // 2)
            )
            conv_layers.append(nn.GELU())
            in_ch = channels
        self.conv_tower = nn.Sequential(*conv_layers)

        self.global_proj = nn.Linear(5, channels)
        self.trunk_proj = nn.Linear(2 * channels, cfg.model_dim)
        self.blocks = nn.ModuleList([
            ResMLPBlock(cfg.model_dim, ff_mult=cfg.ff_mult, dropout=cfg.resid_dropout)
            for _ in range(cfg.n_blocks)
        ])

        delta_hidden = max(cfg.model_dim, int(cfg.model_dim * cfg.delta_hidden_mult))
        self.policy_proj = nn.Linear(cfg.model_dim, delta_hidden)
        self.cand_proj = nn.Linear(cfg.model_dim, delta_hidden)

        value_hidden = max(cfg.model_dim // 2, int(cfg.model_dim * cfg.value_hidden_mult))
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.model_dim),
            nn.Linear(cfg.model_dim, value_hidden),
            nn.GELU(),
            nn.Linear(value_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = np.sqrt(2)
                if module.out_features == 1:
                    gain = 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, boards: torch.Tensor, sec_flags: torch.Tensor) -> torch.Tensor:
        X_points, X_global = encode_board_cnn_torch(boards, sec_flags)
        feats = self.conv_tower(X_points)
        if self.cfg.conv_pool == "max":
            pooled = feats.max(dim=-1).values
        else:
            pooled = feats.mean(dim=-1)
        globals_enc = F.gelu(self.global_proj(X_global))
        trunk = torch.cat([pooled, globals_enc], dim=-1)
        h = self.trunk_proj(trunk)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, states: torch.Tensor, deltas: torch.Tensor, mask: torch.Tensor):
        board = states[..., :self.board_dim]
        sec_flag = states[..., self.flag_index]

        cand_states = states.unsqueeze(1) + deltas
        cand_boards = cand_states[..., :self.board_dim]
        cand_flags = cand_states[..., self.flag_index]

        batch, actions, _ = cand_boards.shape
        state_feat = self._encode(board, sec_flag)
        cand_feat = self._encode(
            cand_boards.reshape(batch * actions, self.board_dim),
            cand_flags.reshape(batch * actions),
        ).view(batch, actions, -1)

        query = self.policy_proj(state_feat).unsqueeze(1)
        keys = self.cand_proj(cand_feat)
        logits = torch.sum(query * keys, dim=-1)
        logits = logits.masked_fill(mask == 0, -1e9)

        values = self.value_head(state_feat).squeeze(-1)
        return logits, values

    def value(self, states: torch.Tensor) -> torch.Tensor:
        board = states[..., :self.board_dim]
        sec_flag = states[..., self.flag_index]
        feats = self._encode(board, sec_flag)
        return self.value_head(feats).squeeze(-1)


class PPOConvTournamentAgent:
    """Inference-only PPO Conv agent packaged for tournaments."""

    def __init__(self, config: ConvConfig | None = None, device: str | torch.device | None = None):
        self.config = config or ConvConfig()
        self.device = _detect_device(str(device) if isinstance(device, torch.device) else device)
        self.acnet = PPOConvActorCritic(self.config).to(self.device)
        self.acnet.eval()

    def set_eval_mode(self, is_eval: bool = True) -> None:
        if is_eval:
            self.acnet.eval()
        else:
            self.acnet.train()

    def load(self, checkpoint_path: str | Path, map_location: str | torch.device | None = None) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        location = map_location if map_location is not None else self.device
        if isinstance(location, str):
            location = torch.device(location)
        checkpoint = torch.load(path, map_location=location)
        state_dict = checkpoint.get("acnet", checkpoint)

        # Torch 2.x can wrap modules inside _orig_mod when using torch.compile.
        prefix = "_orig_mod."
        if isinstance(state_dict, dict) and any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()
            }

        self.acnet.load_state_dict(state_dict)
        print(f"Loaded PPO Conv weights from {path}")

    def _encode_state(self, board29: np.ndarray, moves_left: int = 0) -> np.ndarray:
        feat = np.zeros(self.config.state_dim, dtype=np.float32)
        feat[: self.config.raw_board_dim] = board29.astype(np.float32)
        feat[self.config.second_roll_index] = 1.0 if moves_left > 1 else 0.0
        return feat

    def _encode_batch(self, boards29: List[np.ndarray], moves_left: int = 0) -> torch.Tensor:
        encoded = [self._encode_state(board, moves_left) for board in boards29]
        stacked = np.stack(encoded, axis=0).astype(np.float32)
        return torch.as_tensor(stacked, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _evaluate_moves_lookahead(self, possible_boards: Sequence[np.ndarray], player: int) -> torch.Tensor:
        if not possible_boards:
            return torch.empty(0, device=self.device)

        dice_rolls = [
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6),
            (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6),
        ]
        dice_weights = np.array([1 if a == b else 2 for (a, b) in dice_rolls], dtype=np.float32) / 36.0

        all_opp_states: List[np.ndarray] = []
        move_roll_map: List[Tuple[int, int, int, int]] = []
        my_board_values: List[float] = []

        for move_idx, board_after in enumerate(possible_boards):
            board_after = np.asarray(board_after, dtype=np.float32)
            board_opp_pov = _flip_board(board_after)

            encoded_my_board = self._encode_batch([board_after])
            my_board_values.append(self.acnet.value(encoded_my_board).item())

            for dice_idx, dice in enumerate(dice_rolls):
                _, opp_boards = backgammon.legal_moves(board_opp_pov, dice, player=1)

                start = len(all_opp_states)
                if opp_boards:
                    all_opp_states.extend(np.asarray(b, dtype=np.float32) for b in opp_boards)
                end = len(all_opp_states)
                move_roll_map.append((move_idx, dice_idx, start, end))

        if not all_opp_states:
            return torch.tensor(my_board_values, dtype=torch.float32, device=self.device)

        opp_encoded = self._encode_batch(all_opp_states)
        all_opp_vals = self.acnet.value(opp_encoded)

        final_vals = [0.0] * len(possible_boards)
        for move_idx, dice_idx, start, end in move_roll_map:
            weight = dice_weights[dice_idx]
            if start == end:
                my_val = my_board_values[move_idx]
            else:
                best_reply = torch.max(all_opp_vals[start:end]).item()
                my_val = -best_reply
            final_vals[move_idx] += my_val * float(weight)

        return torch.tensor(final_vals, dtype=torch.float32, device=self.device)

    def action(self, board: np.ndarray, dice: Tuple[int, int], player: int,
               roll_index: int = 0, train_config: dict | None = None) -> List[List[int]]:
        board_arr = np.asarray(board, dtype=np.float32)
        board_pov = _flip_board(board_arr) if player == -1 else board_arr
        possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
        n_actions = len(possible_moves)

        if n_actions == 0:
            return []

        if n_actions == 1:
            chosen = possible_moves[0]
        else:
            moves_left = 1 + int(dice[0] == dice[1]) - roll_index
            S = self._encode_state(board_pov, moves_left)
            cand_states = np.stack(
                [self._encode_state(np.asarray(b, dtype=np.float32), moves_left - 1) for b in possible_boards],
                axis=0,
            )
            deltas = cand_states - S
            mask = np.ones(n_actions, dtype=np.float32)

            S_t = torch.as_tensor(S[None, :], dtype=torch.float32, device=self.device)
            deltas_t = torch.as_tensor(deltas[None, :, :], dtype=torch.float32, device=self.device)
            mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32, device=self.device)

            use_la = False
            la_depth = 1
            if isinstance(train_config, dict):
                use_la = bool(train_config.get("use_lookahead", False))
                la_depth = int(train_config.get("lookahead_k", 1) or 1)

            if use_la and la_depth >= 1:
                if la_depth == 1:
                    logits = self._evaluate_moves_lookahead(possible_boards, player)
                else:
                    with torch.no_grad():
                        logits, _ = self.acnet(S_t, deltas_t, mask_t)
                        logits = logits.squeeze(0)
            else:
                with torch.no_grad():
                    logits, _ = self.acnet(S_t, deltas_t, mask_t)
                    logits = logits.squeeze(0)

            idx = torch.argmax(logits).item()
            chosen = possible_moves[idx]

        if player == -1:
            return _flip_move(chosen)
        return chosen


_AGENT: PPOConvTournamentAgent | None = None
_AGENT_INFO: dict[str, Any] = {}


def initialize(checkpoint_path: str = _DEFAULT_CHECKPOINT,
               device: str | None = None,
               model_size: str = "large") -> PPOConvTournamentAgent:
    global _AGENT, _AGENT_INFO

    checkpoint_str = str(checkpoint_path)
    device_str = str(device) if device is not None else None
    meta = {
        "checkpoint": checkpoint_str,
        "device": device_str,
        "model_size": model_size,
    }

    if (_AGENT is None) or any(_AGENT_INFO.get(k) != v for k, v in meta.items()):
        cfg = get_config(model_size)
        _AGENT = PPOConvTournamentAgent(config=cfg, device=device)
        _AGENT.load(checkpoint_str, map_location=device)
        _AGENT.set_eval_mode(True)
        _AGENT_INFO = meta
    return _AGENT


def action(board,
           dice,
           player,
           roll_index: int | None = None,
           i: int | None = None,
           use_lookahead: bool | None = None,
           **kwargs):
    checkpoint_override = kwargs.pop("checkpoint_path", None)
    device = kwargs.pop("device", None)
    model_size = kwargs.pop("model_size", "large")

    agent = initialize(
        checkpoint_path=checkpoint_override or _DEFAULT_CHECKPOINT,
        device=device,
        model_size=model_size,
    )

    resolved_roll_index: int
    if roll_index is not None and i is not None and roll_index != i:
        resolved_roll_index = roll_index
    elif roll_index is not None:
        resolved_roll_index = roll_index
    elif i is not None:
        resolved_roll_index = i
    else:
        resolved_roll_index = 0

    train_config = kwargs.pop("train_config", None)
    if train_config is None:
        use_la = use_lookahead if use_lookahead is not None else kwargs.pop("use_lookahead", True)
        lookahead_k = kwargs.pop("lookahead_k", 1 if use_la else 0)
        train_config = {"use_lookahead": use_la, "lookahead_k": lookahead_k}
    else:
        train_config = dict(train_config)

    return agent.action(board, dice, player, roll_index=resolved_roll_index, train_config=train_config)


# Auto-initialize so tournament.py can import-and-go
try:
    initialize()
except FileNotFoundError:
    # Allow loading to be deferred if default checkpoint is absent
    pass


__all__ = [
    "PPOConvTournamentAgent",
    "initialize",
    "action",
    "get_config",
]
