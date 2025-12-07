#!/usr/bin/env python3
"""CNN-based PPO agent that mirrors the MLP agent but uses a Conv encoder."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ppo_agent as base_agent


def encode_board_cnn_torch(board29: torch.Tensor, n_second_roll: torch.Tensor):
    """Encode raw 29-dim boards into convolutional channels + globals."""
    if board29.dim() == 1:
        board29 = board29.unsqueeze(0)
    board29 = board29.to(dtype=torch.float32)
    B = board29.size(0)

    points = board29[:, 1:25]
    ours = torch.clamp(points, min=0)
    opp = torch.clamp(-points, min=0)

    # Occupancy bins for each side
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
    idx = idx.view(1, 24).expand(B, -1)
    home = torch.zeros_like(idx)
    home[:, :6] = 1.0

    X_points = torch.stack(
        pos_bins + neg_bins + [idx, home],
        dim=1,
    )

    bar_p = board29[:, 25]
    bar_o = board29[:, 26]
    off_p = board29[:, 27] / 15.0
    off_o = board29[:, 28].abs() / 15.0

    if not torch.is_tensor(n_second_roll):
        n_second_roll = torch.tensor(n_second_roll, device=board29.device, dtype=torch.float32)
    else:
        n_second_roll = n_second_roll.to(device=board29.device, dtype=torch.float32)
    if n_second_roll.dim() == 0:
        n_second_roll = n_second_roll.expand(B)
    n_second_roll = n_second_roll.reshape(B)

    X_global = torch.stack(
        [bar_p, bar_o, off_p, off_o, n_second_roll],
        dim=-1,
    )

    return X_points, X_global


class ConvConfig(base_agent.Config):
    """Config shared by CNN agents."""

    raw_board_dim = 29
    state_dim = raw_board_dim + 1  # append doubles flag
    second_roll_index = raw_board_dim
    use_raw_board_inputs = True

    conv_in_channels = 10
    conv_channels = 64
    conv_kernel_size = 3
    conv_layers = 3
    conv_pool = "mean"


class SmallConvConfig(ConvConfig):
    model_dim = 256
    n_blocks = 4
    conv_channels = 48
    conv_layers = 2
    rollout_length = 256
    minibatch_size = 64
    resid_dropout = 0.05


class MediumConvConfig(ConvConfig):
    model_dim = 384
    n_blocks = 5
    conv_channels = 64
    conv_layers = 3
    rollout_length = 384
    minibatch_size = 96
    resid_dropout = 0.05


class LargeConvConfig(ConvConfig):
    model_dim = 640
    n_blocks = 8
    conv_channels = 96
    conv_layers = 4
    rollout_length = 640
    minibatch_size = 160
    resid_dropout = 0.1
    compile_model = True


def get_config(size: str = 'large') -> ConvConfig:
    configs = {
        'small': SmallConvConfig,
        'medium': MediumConvConfig,
        'large': LargeConvConfig,
    }
    size_norm = size.lower()
    if size_norm not in configs:
        raise ValueError(f"Unknown CNN size '{size}'. Choose from {list(configs.keys())}.")

    cfg = configs[size_norm]()
    print(f"\nCNN Model Configuration: {size_norm.upper()}")
    print(f"  Conv channels: {cfg.conv_channels} x {cfg.conv_layers} layers (k={cfg.conv_kernel_size})")
    print(f"  Trunk dim: {cfg.model_dim}, blocks={cfg.n_blocks}, ff_mult={cfg.ff_mult}")
    print(f"  Rollout: length={cfg.rollout_length}, batch={cfg.minibatch_size}")
    print(f"  Learning rate: {cfg.lr}")
    return cfg


CFG = ConvConfig()


class PPOConvActorCritic(nn.Module):
    """Actor-critic with CNN encoder for state and candidate boards."""

    def __init__(self, cfg: ConvConfig):
        super().__init__()
        self.cfg = cfg
        self.board_dim = getattr(cfg, 'raw_board_dim', 29)
        self.flag_index = getattr(cfg, 'second_roll_index', self.board_dim)

        C = cfg.conv_channels
        conv_layers = []
        in_ch = cfg.conv_in_channels
        for _ in range(cfg.conv_layers):
            conv_layers.append(
                nn.Conv1d(in_ch, C, kernel_size=cfg.conv_kernel_size, padding=cfg.conv_kernel_size // 2)
            )
            conv_layers.append(nn.GELU())
            in_ch = C
        self.conv_tower = nn.Sequential(*conv_layers)

        self.global_proj = nn.Linear(5, C)
        self.trunk_proj = nn.Linear(2 * C, cfg.model_dim)
        self.blocks = nn.ModuleList([
            base_agent.ResMLPBlock(cfg.model_dim, ff_mult=cfg.ff_mult, dropout=cfg.resid_dropout)
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

    def _init_weights(self):
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

    def _encode(self, boards, sec_flags):
        X_points, X_global = encode_board_cnn_torch(boards, sec_flags)
        feats = self.conv_tower(X_points)
        if self.cfg.conv_pool == 'max':
            pooled = feats.max(dim=-1).values
        else:
            pooled = feats.mean(dim=-1)
        globals_enc = F.gelu(self.global_proj(X_global))
        trunk = torch.cat([pooled, globals_enc], dim=-1)
        h = self.trunk_proj(trunk)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, states, deltas, mask):
        board = states[..., :self.board_dim]
        sec_flag = states[..., self.flag_index]

        cand_states = states.unsqueeze(1) + deltas
        cand_boards = cand_states[..., :self.board_dim]
        cand_flags = cand_states[..., self.flag_index]

        B, A, _ = cand_boards.shape
        state_feat = self._encode(board, sec_flag)
        cand_feat = self._encode(
            cand_boards.reshape(B * A, self.board_dim),
            cand_flags.reshape(B * A),
        ).view(B, A, -1)

        query = self.policy_proj(state_feat).unsqueeze(1)
        keys = self.cand_proj(cand_feat)
        logits = torch.sum(query * keys, dim=-1)
        logits = logits.masked_fill(mask == 0, -1e9)

        values = self.value_head(state_feat).squeeze(-1)
        return logits, values

    def value(self, states):
        """Compute value head output for encoded states."""
        board = states[..., :self.board_dim]
        sec_flag = states[..., self.flag_index]
        feats = self._encode(board, sec_flag)
        return self.value_head(feats).squeeze(-1)


class PPOConvAgent(base_agent.PPOAgent):
    """PPO agent that swaps in the CNN actor-critic."""

    def __init__(self, config: ConvConfig = None, device=None,
                 teacher_mode: str = 'pubeval', teacher_module=None):
        cfg = config or CFG
        super().__init__(config=cfg, device=device,
                         teacher_mode=teacher_mode, teacher_module=teacher_module)
        self.config.use_raw_board_inputs = True
        self.uses_raw_board_state = True

        self.acnet = PPOConvActorCritic(self.config).to(self.device)
        if getattr(torch, "compile", None) and getattr(self.config, "compile_model", False):
            try:
                self.acnet = torch.compile(self.acnet)
                print("  torch.compile enabled for PPOConvActorCritic")
            except Exception as compile_err:
                print(f"  torch.compile unavailable for CNN agent ({compile_err}); continuing without it.")

        self.optimizer = torch.optim.AdamW(
            self.acnet.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self._base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def _encode_state(self, board29, moves_left=0):
        feat = np.zeros(self.config.state_dim, dtype=np.float32)
        feat[:self.config.raw_board_dim] = board29.astype(np.float32)
        feat[self.config.second_roll_index] = 1.0 if moves_left > 1 else 0.0
        return feat

    def batch_score(self, states, cand_states, masks):
        S_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        C_t = torch.as_tensor(cand_states, dtype=torch.float32, device=self.device)
        deltas = C_t - S_t[:, None, :]
        M_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        return self.acnet(S_t, deltas, M_t)


__all__ = [
    'ConvConfig', 'SmallConvConfig', 'MediumConvConfig', 'LargeConvConfig',
    'get_config', 'PPOConvAgent'
]
