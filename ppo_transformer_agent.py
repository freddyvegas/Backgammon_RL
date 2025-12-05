from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from utils import (
    _flip_board,
    _flip_move,
    transformer_one_hot_encoding,
    transformer_one_hot_encoding_torch,
    transformer_start_token,
    pad_truncate_seq,
    get_device,
    TRANSFORMER_STATE_DIM,
)
import backgammon

"""
Transformer-based PPO Agent for Backgammon
-----------------------------------------

This module replaces the ResMLP actor-critic with a decoder-only Transformer.
The Transformer consumes a sequence of tokens representing the *history of game
states* from the current player's POV (optionally preceded by a learnable BOS
token). The model outputs a policy over candidate next-moves and a value
estimate for the current state.

Key design:
- State encoder: MLP mapping 293-dim one-hot features -> d_model.
- Candidate encoder: MLP mapping (delta = candidate_onehot - current_onehot) -> d_model.
- Decoder-only Transformer with causal attention over the history tokens.
- Policy head: dot-product between the final token's hidden state and candidate
  embeddings (masked by legal moves).
- Value head: MLP on the final token's hidden state.
- PPO training loop maintained with minimal changes; rollout buffer stores
  sequences (padded/truncated to max_seq_len) + candidates + masks.
- Teacher signal (pubeval) is preserved as optional DAGGER-lite supervision.

Notes:
- The module-level API (save/load/action/episode_start/end_episode/etc.) is kept
  so this can drop in for the prior PPO agent.
- Warmstart with pubeval is kept (converted to sequence form with a single-token
  sequence by default).
"""

# ---------------- Config ----------------
class Config:
    """Base configuration with Transformer hyperparameters and PPO settings."""
    # Input feature dim (augmented transformer tokens)
    state_dim = TRANSFORMER_STATE_DIM
    max_actions = 64

    # Transformer / tokenization
    max_seq_len = 64              # N: use last N tokens from history
    use_bos_token = True          # Learnable BOS token at start of sequence
    limit_history_to_max_seq = True  # Clip env histories to avoid unbounded growth
    vectorize_candidate_encoding = True  # Encode legal moves in a single tensor op

    # Transformer sizes (LARGE default)
    d_model = 512
    n_layers = 8
    n_heads = 8
    d_ff = 2048
    attn_dropout = 0.1
    resid_dropout = 0.1

    # PPO hyperparameters (gentler for stability)
    lr = 5e-5
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.1
    # Learning-rate schedule (legacy params kept for parity with PPO MLP agent)
    lr_warmup_updates = 100
    lr_cosine_updates = 3000
    lr_min_ratio = 0.1

    # Exploration
    entropy_coef = 0.02
    entropy_decay = 0.9999
    entropy_min = 0.0

    critic_coef = 1.0
    eval_temperature = 0.01

    # PPO rollout settings
    rollout_length = 512
    ppo_epochs = 2
    minibatch_size = 128

    # Gradient clipping
    grad_clip = 0.5
    max_grad_norm = 0.5

    # Weight decay
    weight_decay = 1e-6

    # Reward scaling
    reward_scale = 1.0

    # Reward shaping (DISABLED)
    use_reward_shaping = False
    pip_reward_scale = 0.001
    bear_off_reward = 0.01
    hit_reward = 0.01
    shaping_clip = 0.05

    # Teacher signal (DAGGER-lite)
    teacher_sample_rate = 0.10
    teacher_loss_coef_start = 0.05
    teacher_loss_coef_end = 0.0
    teacher_decay_horizon = 50_000

    # Compilation / performance knobs
    compile_model = False


class SmallConfig(Config):
    """Small model for CPU training / quick tests."""
    d_model = 192
    n_layers = 4
    n_heads = 4
    d_ff = 768
    max_seq_len = 32
    rollout_length = 256
    minibatch_size = 64
    ppo_epochs = 1
    clip_epsilon = 0.1


class MediumConfig(Config):
    """Medium model for balanced training."""
    d_model = 320
    n_layers = 6
    n_heads = 5
    d_ff = 1280
    max_seq_len = 48
    rollout_length = 384
    minibatch_size = 96


class LargeConfig(Config):
    """Large model (default)."""
    d_model = 640
    n_layers = 10
    n_heads = 10
    d_ff = 2560
    max_seq_len = 64
    rollout_length = 640
    minibatch_size = 160
    ppo_epochs = 2
    compile_model = True


def get_config(size='large'):
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
    print(f"  Transformer: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, ff={cfg.d_ff}")
    print(f"  Max seq len: {cfg.max_seq_len}  | BOS: {cfg.use_bos_token}")
    print(f"  Rollout: length={cfg.rollout_length}, batch={cfg.minibatch_size}")
    print(f"  Learning rate: {cfg.lr} (gentler)")
    print(f"  PPO clip ε: {cfg.clip_epsilon} (gentler)")
    print(f"  PPO epochs: {cfg.ppo_epochs} (gentler)")
    print(f"  Teacher signal: {cfg.teacher_sample_rate*100:.0f}% sample rate")
    print(f"  torch.compile: {cfg.compile_model}")
    return cfg


# Default config instance
CFG = Config()

# Set device at module level
CFG.device = get_device()
print(f"Transformer PPO agent using device: {CFG.device}")
if CFG.device == "mps":
    print("  ⚠️  Note: MPS may have stability issues with PPO. Consider using CPU.")


# ------------- Rollout Buffer (sequence-aware) -------------
class PPORolloutBuffer:
    """Stores on-policy rollouts for PPO updates (sequence version)."""
    def __init__(self, rollout_length=512):
        self.rollout_length = rollout_length
        self.clear()

    def clear(self):
        self.seqs = []            # (L, 293) padded/truncated sequences
        self.seq_lens = []        # true lengths before padding (int)
        self.candidate_states = []
        self.masks = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.teacher_indices = []
        self.size = 0

    def push(self, seq, seq_len, candidate_states, mask, action, log_prob, value, reward, done, teacher_idx=-1):
        self.seqs.append(seq)
        self.seq_lens.append(seq_len)
        self.candidate_states.append(candidate_states)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.teacher_indices.append(teacher_idx)
        self.size += 1

    def is_ready(self):
        return self.size >= self.rollout_length

    def get(self):
        return (
            np.array(self.seqs, dtype=np.float32),              # (T, L, 293)
            np.array(self.seq_lens, dtype=np.int64),            # (T,)
            np.array(self.candidate_states, dtype=np.float32),  # (T, A, 293)
            np.array(self.masks, dtype=np.float32),             # (T, A)
            np.array(self.actions, dtype=np.int64),             # (T,)
            np.array(self.log_probs, dtype=np.float32),         # (T,)
            np.array(self.values, dtype=np.float32),            # (T,)
            np.array(self.rewards, dtype=np.float32),           # (T,)
            np.array(self.dones, dtype=np.float32),             # (T,)
            np.array(self.teacher_indices, dtype=np.int64),     # (T,)
        )


# ------------- Transformer Blocks -------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(resid_dropout),
        )

    def forward(self, x, attn_mask):
        # x: (B, L, d)
        h = self.ln1(x)
        # attn_mask: (L, L) with True for disallowed positions (causal)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerActorCritic(nn.Module):
    """
    Decoder-only Transformer that produces:
      - logits over candidate moves from the final token's hidden state
      - scalar value estimate from the final token's hidden state
    """
    def __init__(self, state_dim=293, d_model=512, n_layers=8, n_heads=8, d_ff=2048,
                 max_seq_len=64, use_bos_token=True, resid_dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_bos_token = use_bos_token

        # Token encoder for states (293 -> d_model)
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Candidate delta encoder (293 -> d_model)
        self.delta_enc = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional embeddings
        pos_tokens = max_seq_len + int(use_bos_token)
        self.pos_emb = nn.Parameter(torch.zeros(1, pos_tokens, d_model))

        # Optional BOS token (learnable)
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model)) if use_bos_token else None

        # Pre-computed causal mask (trimmed at runtime)
        causal = torch.triu(torch.ones(pos_tokens, pos_tokens, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask_cache", causal, persistent=False)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_emb, std=0.01)
        if self.bos is not None:
            nn.init.normal_(self.bos, std=0.02)

    def _causal_mask(self, L: int, device: torch.device):
        if hasattr(self, "causal_mask_cache") and self.causal_mask_cache.size(0) >= L:
            return self.causal_mask_cache[:L, :L]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def _encode_sequence(self, seq_feats: torch.Tensor, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of sequences.
        Args:
            seq_feats: (B, L, 293) padded features
            seq_lens:  (B,) true lengths (<= L)
        Returns:
            h_all: (B, L', d) hidden states (including BOS if enabled)
            last_idx: (B,) index of final (non-pad) token per sequence in h_all
        """
        B, L, D = seq_feats.shape
        device = seq_feats.device

        x = self.state_enc(seq_feats)  # (B, L, d_model)

        if self.use_bos_token:
            bos = self.bos.expand(B, 1, -1)
            x = torch.cat([bos, x], dim=1)  # (B, 1+L, d)
            pos = self.pos_emb[:, : (L + 1), :]
            last_idx = seq_lens + 0  # last token index shifts by 1 for BOS? We want last=seq_lens (index of last real token)
        else:
            pos = self.pos_emb[:, : L, :]
            last_idx = seq_lens - 1

        x = x + pos

        Lp = x.size(1)
        attn_mask = self._causal_mask(Lp, device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = self.ln_f(x)
        return x, last_idx

    def forward(self, seq_feats, seq_lens, cand_feats, mask):
        """
        Args:
            seq_feats: (B, L, 293)  sequence of state features (padded)
            seq_lens:  (B,)         true lengths (<= L)
            cand_feats: (B, A, 293) candidate one-hot features (after-state)
            mask:      (B, A)       1 for valid candidate, 0 for padding

        Returns:
            logits: (B, A) policy logits (masked)
            values: (B,)  value estimates
        """
        B, L, _ = seq_feats.shape
        device = seq_feats.device

        # Encode sequence and gather final hidden state for each sample
        h_all, last_idx = self._encode_sequence(seq_feats, seq_lens)
        # Adjust last_idx for BOS if used
        if self.use_bos_token:
            # last index equals seq_lens (since BOS occupies 0)
            idx = last_idx.clamp(min=1)  # safety
        else:
            idx = last_idx.clamp(min=0)
        gather_idx = idx.view(B, 1, 1).expand(B, 1, self.d_model)
        h_last = h_all.gather(dim=1, index=gather_idx).squeeze(1)  # (B, d)

        # Candidate deltas: cand - current (current is last feature in each sequence)
        # Recover the current state feature (last real token) from seq_feats
        cur_idx = (seq_lens - 1).clamp(min=0)
        cur_idx_expand = cur_idx.view(B, 1, 1).expand(B, 1, seq_feats.size(-1))
        cur_state = seq_feats.gather(dim=1, index=cur_idx_expand).squeeze(1)  # (B, 293)

        deltas = cand_feats - cur_state.unsqueeze(1)  # (B, A, 293)
        delta_emb = self.delta_enc(deltas)            # (B, A, d)

        # Dot-product scoring
        logits = torch.sum(delta_emb * h_last.unsqueeze(1), dim=-1)  # (B, A)

        # Mask invalid candidates
        logits = logits.masked_fill(mask == 0, -1e9)

        # Value head from last hidden state
        values = self.value_head(h_last).squeeze(-1)
        return logits, values


# ------------- PPO Agent -------------
class PPOAgent:
    def __init__(self, config=None, device=None,
                 teacher_mode: str = 'pubeval', teacher_module=None):
        self.config = config or CFG
        self.device = device or get_device()
        mode = (teacher_mode or 'none').lower()
        if mode not in ('pubeval', 'gnubg', 'none'):
            print(f"Unknown teacher '{teacher_mode}', defaulting to 'none'.")
            mode = 'none'
        self.teacher_type = mode
        self.teacher_module = teacher_module if mode != 'none' else None

        self.acnet = TransformerActorCritic(
            state_dim=self.config.state_dim,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            use_bos_token=self.config.use_bos_token,
            resid_dropout=self.config.resid_dropout,
            attn_dropout=self.config.attn_dropout,
        ).to(self.device)

        if getattr(torch, "compile", None) and getattr(self.config, "compile_model", False):
            try:
                self.acnet = torch.compile(self.acnet)
                print("  torch.compile enabled for TransformerActorCritic")
            except Exception as compile_err:
                print(f"  torch.compile unavailable ({compile_err}); continuing without it.")

        self.optimizer = torch.optim.AdamW(
            self.acnet.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self._base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.training_horizon = None

        self.buffer = PPORolloutBuffer(rollout_length=self.config.rollout_length)

        self.steps = 0
        self.updates = 0
        self.eval_mode = False
        self.current_entropy_coef = self.config.entropy_coef

        # History of POV states (29-dim raw) and features (293-dim one-hot) for the current episode
        self._history_states29: List[np.ndarray] = []
        self._history_feats293: List[np.ndarray] = []
        self._history_has_start: bool = True
        self._start_token_np = transformer_start_token()

        self.rollout_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'teacher_loss': [],
            'masked_entropy': [],
            'grad_norm': [],
            'nA_values': [],
        }

        print("Transformer PPOAgent initialized:")
        print(f"  Device: {self.device}")
        print(f"  LR: {self.config.lr} (gentler)")
        print(f"  Transformer: d={self.config.d_model}, L={self.config.n_layers}, H={self.config.n_heads}, FF={self.config.d_ff}")
        print(f"  Max seq: {self.config.max_seq_len}, BOS={self.config.use_bos_token}")
        teacher_label = self.teacher_type if self.teacher_module is not None else 'disabled'
        print(f"  Teacher source: {teacher_label}")

    def _lr_scale(self, game_count: int) -> float:
        horizon = float(self.training_horizon or 1)
        progress = min(max(game_count / horizon, 0.0), 1.0)
        warmup = float(getattr(self.config, 'lr_warmup_ratio', 0.1))
        if warmup > 0 and progress < warmup:
            return max(1e-3, progress / warmup)
        if warmup > 0:
            progress = (progress - warmup) / max(1e-9, 1.0 - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.config.lr_min_ratio, cosine)

    def set_training_horizon(self, total_games: int):
        if total_games is None:
            return
        self.training_horizon = max(1, int(total_games))
        self.update_lr_schedule(0)

    def update_lr_schedule(self, games_done: int, total_games: int = None):
        if total_games is not None:
            self.set_training_horizon(total_games)
        if not self._base_lrs:
            return
        scale = self._lr_scale(int(max(0, games_done)))
        for lr, group in zip(self._base_lrs, self.optimizer.param_groups):
            group['lr'] = lr * scale

    def _teacher_enabled(self) -> bool:
        return (
            self.teacher_module is not None and
            self.teacher_type != 'none' and
            self.config.teacher_sample_rate > 0
        )

    def has_teacher(self) -> bool:
        return self._teacher_enabled()

    # -------- Utility encoders and rewards --------
    def _encode_state(self, board29, moves_left=0, actor_flag=0.0):
        return transformer_one_hot_encoding(board29.astype(np.float32), bool(moves_left > 1), actor_flag)

    def _compute_shaped_reward(self, board_before, board_after):
        if not self.config.use_reward_shaping:
            return 0.0
        pip_before = np.sum(np.arange(1, 25) * np.maximum(board_before[1:25], 0))
        pip_after = np.sum(np.arange(1, 25) * np.maximum(board_after[1:25], 0))
        pip_reward = -self.config.pip_reward_scale * (pip_after - pip_before)
        bear_off_reward = self.config.bear_off_reward * (board_after[27] - board_before[27])
        hit_reward = self.config.hit_reward * (board_after[26] - board_before[26])
        total_shaped = pip_reward + bear_off_reward + hit_reward
        return float(np.clip(total_shaped, -self.config.shaping_clip, self.config.shaping_clip))

    # -------- Checkpointing --------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'acnet': self.acnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'entropy_coef': self.current_entropy_coef,
            'config': {
                'state_dim': self.config.state_dim,
                'd_model': self.config.d_model,
                'n_layers': self.config.n_layers,
                'n_heads': self.config.n_heads,
                'd_ff': self.config.d_ff,
                'max_seq_len': self.config.max_seq_len,
                'use_bos_token': self.config.use_bos_token,
                'lr': self.config.lr,
                'clip_epsilon': self.config.clip_epsilon,
                'ppo_epochs': self.config.ppo_epochs,
            }
        }, path)

    def load(self, path: str, map_location: Union[str, torch.device] = None, load_optimizer: bool = True):
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        saved_config = checkpoint.get('config', {})
        if saved_config:
            arch_diff = (
                saved_config.get('state_dim', self.config.state_dim) != self.config.state_dim or
                saved_config.get('d_model', self.config.d_model) != self.config.d_model or
                saved_config.get('n_layers', self.config.n_layers) != self.config.n_layers or
                saved_config.get('n_heads', self.config.n_heads) != self.config.n_heads or
                saved_config.get('d_ff', self.config.d_ff) != self.config.d_ff or
                saved_config.get('max_seq_len', self.config.max_seq_len) != self.config.max_seq_len or
                saved_config.get('use_bos_token', self.config.use_bos_token) != self.config.use_bos_token
            )
            if arch_diff:
                print("  Rebuilding Transformer with saved architecture:")
                self.acnet = TransformerActorCritic(
                    state_dim=saved_config.get('state_dim', self.config.state_dim),
                    d_model=saved_config.get('d_model', self.config.d_model),
                    n_layers=saved_config.get('n_layers', self.config.n_layers),
                    n_heads=saved_config.get('n_heads', self.config.n_heads),
                    d_ff=saved_config.get('d_ff', self.config.d_ff),
                    max_seq_len=saved_config.get('max_seq_len', self.config.max_seq_len),
                    use_bos_token=saved_config.get('use_bos_token', self.config.use_bos_token),
                    resid_dropout=self.config.resid_dropout,
                    attn_dropout=self.config.attn_dropout,
                ).to(self.device)
            # Sync current config with saved architecture
            for key, default in [
                ('state_dim', self.config.state_dim),
                ('d_model', self.config.d_model),
                ('n_layers', self.config.n_layers),
                ('n_heads', self.config.n_heads),
                ('d_ff', self.config.d_ff),
                ('max_seq_len', self.config.max_seq_len),
                ('use_bos_token', self.config.use_bos_token),
            ]:
                if key in saved_config:
                    setattr(self.config, key, saved_config[key])
        self.acnet.load_state_dict(checkpoint['acnet'])
        if load_optimizer and 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint.get('steps', 0)
        self.updates = checkpoint.get('updates', 0)
        self.current_entropy_coef = checkpoint.get('entropy_coef', self.config.entropy_coef)
        print(f"Loaded checkpoint: {path}")
        print(f"  Steps: {self.steps:,}")
        print(f"  Updates: {self.updates:,}")

    # -------- Modes --------
    def set_eval_mode(self, is_eval: bool):
        self.eval_mode = is_eval
        self.acnet.eval() if is_eval else self.acnet.train()

    # -------- Batch scoring (compatibility with original agent) --------
    def batch_score(self, states, cand_states, masks, *,
                    histories293: Optional[np.ndarray] = None,
                    histories_len: Optional[np.ndarray] = None,
                    histories29: Optional[list] = None,
                    moves_left_flags: Optional[list] = None):
        """
        Sequence-aware batch scoring (compatible with the original API, but
        upgraded to use *real histories* when provided).

        Args:
            states:       (B, 29) raw +1 POV current boards (float/int) or (B, 293)
                          torch tensor or numpy array. Used only if we need to
                          infer the current token or when histories are missing.
            cand_states:  (B, A, 29) raw +1 POV candidate after-states OR
                          (B, A, 293) pre-encoded features.
            masks:        (B, A) 1.0 for valid, 0.0 for padded (numpy or torch).
            histories293: Optional (B, L, 293) array/tensor of per-sample state
                          features. If given, these are used as the token sequences.
            histories_len:Optional (B,) true lengths for histories293 before padding.
            histories29:  Optional list/array of Python lists, where each element is
                          a sequence of raw 29-d states. If provided (and histories293
                          is None), we will encode them to 293 using one_hot_encoding_torch.
            moves_left_flags: Optional list parallel to histories29 providing a list
                          of bools per step that indicates second-roll flag when
                          encoding (defaults to False everywhere).

        Returns:
            logits: (B, A)
            values: (B,)
        """
        if states is None:
            raise ValueError("batch_score expects 'states' to be provided (for zero-length safety fallback).")

        # ---- Batch size ----
        if torch.is_tensor(states):
            B = states.size(0)
        else:
            B = len(states)

        # ---- Candidate features (C_feat) ----
        # Accept:
        #   - torch or numpy
        #   - either 29-d raw boards, or 293-d encoded features
        if torch.is_tensor(cand_states):
            cand_t = cand_states.to(self.device, dtype=torch.float32)
            if cand_t.shape[-1] != self.config.state_dim:
                actor_val = torch.ones(cand_t.shape[:-1], dtype=torch.float32, device=self.device)
                cand_t = transformer_one_hot_encoding_torch(cand_t, nSecondRoll=False, actor_flag=actor_val)
        else:
            cand_np = np.asarray(cand_states, dtype=np.float32)
            cand_t = torch.from_numpy(cand_np).to(self.device)
            if cand_t.shape[-1] != self.config.state_dim:
                actor_val = torch.ones(cand_t.shape[:-1], dtype=torch.float32, device=self.device)
                cand_t = transformer_one_hot_encoding_torch(cand_t, nSecondRoll=False, actor_flag=actor_val)

        # ---- Masks ----
        if torch.is_tensor(masks):
            mask_t = masks.to(self.device, dtype=torch.float32)
        else:
            mask_np = np.asarray(masks, dtype=np.float32)
            mask_t = torch.from_numpy(mask_np).to(self.device)

        # ---- Build sequence features (seq_pad, true_lens) ----
        N = self.config.max_seq_len
        D = self.config.state_dim

        # 1) histories293 provided (already encoded)
        if histories293 is not None:
            if torch.is_tensor(histories293):
                H_t = histories293.to(self.device, dtype=torch.float32)
            else:
                H_np = np.asarray(histories293, dtype=np.float32)
                H_t = torch.from_numpy(H_np).to(self.device)

            if histories_len is None:
                Lfull = H_t.size(1)
                len_t = torch.full((B,), Lfull, dtype=torch.long, device=self.device)
            else:
                if torch.is_tensor(histories_len):
                    len_t = histories_len.to(self.device, dtype=torch.long)
                else:
                    len_np = np.asarray(histories_len, dtype=np.int64)
                    len_t = torch.from_numpy(len_np).to(self.device)

            seq_pad_t = torch.zeros((B, N, D), dtype=torch.float32, device=self.device)
            true_lens_t = torch.zeros((B,), dtype=torch.long, device=self.device)

            for i in range(B):
                Li = int(len_t[i].item())
                take = min(Li, N)
                if take > 0:
                    seq_slice = H_t[i, Li - take: Li, :]  # last 'take' tokens
                    seq_pad_t[i, :take, :] = seq_slice
                true_lens_t[i] = take

        # 2) histories29 provided: encode sequences on the fly with torch
        elif histories29 is not None:
            seq_pad_t = torch.zeros((B, N, D), dtype=torch.float32, device=self.device)
            true_lens_t = torch.zeros((B,), dtype=torch.long, device=self.device)

            for i in range(B):
                seq29 = histories29[i]
                if seq29 is None or len(seq29) == 0:
                    true_lens_t[i] = 0
                    continue

                Li = len(seq29)
                take = min(Li, N)
                start = Li - take

                # (take, 29) board sequence
                boards29_t = torch.as_tensor(
                    seq29[start:Li],
                    dtype=torch.float32,
                    device=self.device
                )

                # second-roll flags for this sequence (take,)
                if moves_left_flags is not None and i < len(moves_left_flags):
                    flags_i = moves_left_flags[i][start:Li]
                    flags_t = torch.as_tensor(flags_i, dtype=torch.float32, device=self.device)
                else:
                    flags_t = torch.zeros(take, dtype=torch.float32, device=self.device)

                actor_zero = torch.zeros_like(flags_t)
                tokens_t = transformer_one_hot_encoding_torch(boards29_t, flags_t, actor_zero)
                seq_pad_t[i, :take, :] = tokens_t
                true_lens_t[i] = take

        # 3) Fallback: single-token sequences from current state
        else:
            # states may already be 293-d features
            if torch.is_tensor(states):
                S_t = states.to(self.device, dtype=torch.float32)
                if S_t.shape[-1] != D:
                    zeros_flag = torch.zeros(S_t.shape[:-1], dtype=torch.float32, device=self.device)
                    S_t = transformer_one_hot_encoding_torch(S_t, nSecondRoll=False, actor_flag=zeros_flag)
            else:
                states_np = np.asarray(states, dtype=np.float32)
                states_t = torch.from_numpy(states_np).to(self.device)
                if states_t.shape[-1] == D:
                    S_t = states_t
                else:
                    zeros_flag = torch.zeros(states_t.shape[:-1], dtype=torch.float32, device=self.device)
                    S_t = transformer_one_hot_encoding_torch(states_t, nSecondRoll=False, actor_flag=zeros_flag)

            seq_pad_t = torch.zeros((B, N, D), dtype=torch.float32, device=self.device)
            seq_pad_t[:, 0, :] = S_t
            true_lens_t = torch.ones((B,), dtype=torch.long, device=self.device)

        # ---- Safety: ensure min length at least 1 ----
        # If length is 0, inject current state as token 0.
        zero_len_mask = (true_lens_t == 0)
        if zero_len_mask.any():
            # Build a state feature tensor for those problematic samples
            if torch.is_tensor(states):
                states_t = states.to(self.device, dtype=torch.float32)
            else:
                states_np = np.asarray(states, dtype=np.float32)
                states_t = torch.from_numpy(states_np).to(self.device)

            if states_t.shape[-1] == D:
                S_all = states_t
            else:
                zeros_flag = torch.zeros(states_t.shape[:-1], dtype=torch.float32, device=self.device)
                S_all = transformer_one_hot_encoding_torch(states_t, nSecondRoll=False, actor_flag=zeros_flag)

            idxs = torch.nonzero(zero_len_mask, as_tuple=False).view(-1)
            seq_pad_t[idxs, 0, :] = S_all[idxs]
            true_lens_t[zero_len_mask] = 1

        # ---- Forward pass ----
        with torch.no_grad():
            logits, values = self.acnet(seq_pad_t, true_lens_t, cand_t, mask_t)

        return logits, values

    # -------- GAE --------
    def _compute_gae(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if t == T - 1 or dones[t] else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns

    # -------- Teacher helpers --------
    def _get_teacher_label_pubeval(self, state29, after_states29, mask_row):
        if self.teacher_type != 'pubeval' or self.teacher_module is None:
            return -1
        try:
            cand = np.asarray(after_states29, dtype=np.float32)
            nA = min(cand.shape[0], int(mask_row.sum()))
            if nA <= 0:
                return -1
            scores = []
            for k in range(nA):
                after = cand[k]
                race = int(self.teacher_module.israce(after))
                pb28 = self.teacher_module.pubeval_flip(after)
                pos = self.teacher_module._to_int32_view(pb28)
                scores.append(float(self.teacher_module._pubeval_scalar(race, pos)))
            return int(np.argmax(scores))
        except Exception:
            return -1

    def _get_teacher_label_gnubg(self, board_abs, dice, player, pmoves):
        if self.teacher_type != 'gnubg' or self.teacher_module is None:
            return -1
        try:
            move = self.teacher_module.action(board_abs.copy(), np.array(dice, dtype=np.int32), player, 0)
        except Exception:
            return -1
        if move is None or len(move) == 0:
            return -1
        move_arr = np.asarray(move)
        if move_arr.ndim == 1:
            move_arr = move_arr.reshape(1, 2)
        for idx, cand in enumerate(pmoves):
            cand_arr = np.asarray(cand)
            if cand_arr.shape == move_arr.shape and np.array_equal(cand_arr, move_arr):
                return idx
        return -1

    def compute_teacher_index(self, board_abs, dice, player, board_pov, after_states_pov, pmoves, mask):
        if not self._teacher_enabled():
            return -1
        if self.teacher_type == 'pubeval':
            if board_pov is None or after_states_pov is None or mask is None:
                return -1
            mask_row = np.asarray(mask, dtype=np.float32)
            mask_row = mask_row[:after_states_pov.shape[0]]
            idx = self._get_teacher_label_pubeval(board_pov, after_states_pov, mask_row)
            if idx < 0 or idx >= mask_row.shape[0] or mask_row[idx] <= 0:
                return -1
            return idx
        if self.teacher_type == 'gnubg':
            idx = self._get_teacher_label_gnubg(board_abs, dice, player, pmoves)
            if mask is not None:
                mask_row = np.asarray(mask, dtype=np.float32)
                if idx < 0 or idx >= mask_row.shape[0] or mask_row[idx] <= 0:
                    return -1
            return idx
        return -1

    # -------- PPO update --------
    def _ppo_update(self):
        seqs, seq_lens, cand_states, masks, actions, old_log_probs, values, rewards, dones, teacher_indices = self.buffer.get()

        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Tensors
        seqs_t = torch.as_tensor(seqs, dtype=torch.float32, device=self.device)
        seq_lens_t = torch.as_tensor(seq_lens, dtype=torch.int64, device=self.device)
        cand_states_t = torch.as_tensor(cand_states, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        teacher_indices_t = torch.as_tensor(teacher_indices, dtype=torch.int64, device=self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_teacher_loss = 0.0
        n_updates = 0

        dataset_size = len(seqs)
        indices = np.arange(dataset_size)

        for epoch in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, self.config.minibatch_size):
                end_idx = min(start_idx + self.config.minibatch_size, dataset_size)
                mb = indices[start_idx:end_idx]

                mb_seq = seqs_t[mb]
                mb_len = seq_lens_t[mb]
                mb_cand = cand_states_t[mb]
                mb_mask = masks_t[mb]
                mb_act = actions_t[mb]
                mb_oldlp = old_log_probs_t[mb]
                mb_adv = advantages_t[mb]
                mb_ret = returns_t[mb]
                mb_oldv = old_values_t[mb]
                mb_teacher = teacher_indices_t[mb]

                logits, value_preds = self.acnet(mb_seq, mb_len, mb_cand, mb_mask)

                log_probs_all = F.log_softmax(logits.masked_fill(mb_mask == 0, -1e9), dim=-1)
                log_probs = log_probs_all.gather(1, mb_act.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(log_probs - mb_oldlp)
                ratio = torch.clamp(ratio, 0.5, 2.0)  # extra clamp for stability

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_t = value_preds
                v_t_old = mb_oldv
                v_t_clipped = v_t_old + torch.clamp(v_t - v_t_old, -self.config.clip_epsilon, self.config.clip_epsilon)
                v_unclipped = (v_t - mb_ret).pow(2)
                v_clipped = (v_t_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                probs = F.softmax(logits.masked_fill(mb_mask == 0, -1e9), dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                # Teacher loss
                max_actions = mb_mask.size(1)
                valid_teacher = (mb_teacher >= 0) & (mb_teacher < max_actions)
                if valid_teacher.any():
                    safe_idx = mb_teacher.clamp(0, max_actions - 1)
                    teacher_lp = log_probs_all.gather(1, safe_idx.unsqueeze(1)).squeeze(1)
                    teacher_loss = -teacher_lp[valid_teacher].mean()
                else:
                    teacher_loss = torch.tensor(0.0, device=self.device)

                # Cosine decay for teacher coefficient
                alpha0 = self.config.teacher_loss_coef_start
                horizon = self.config.teacher_decay_horizon
                alpha = alpha0 * max(0.0, 0.5 * (1 + math.cos(math.pi * min(self.steps, horizon) / horizon)))

                loss = (policy_loss + self.config.critic_coef * value_loss - self.current_entropy_coef * entropy + alpha * teacher_loss)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_teacher_loss += teacher_loss.item()
                n_updates += 1

        if n_updates > 0:
            self.rollout_stats['policy_loss'].append(total_policy_loss / n_updates)
            self.rollout_stats['value_loss'].append(total_value_loss / n_updates)
            self.rollout_stats['entropy'].append(total_entropy / n_updates)
            self.rollout_stats['teacher_loss'].append(total_teacher_loss / n_updates)
            self.rollout_stats['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        self.current_entropy_coef = max(self.config.entropy_min, self.current_entropy_coef * self.config.entropy_decay)
        self.updates += 1

        if self.updates % 10 == 0:
            samples_per_update = self.config.rollout_length * self.config.ppo_epochs
            print(f"\n[PPO Update #{self.updates}]")
            print(f"  Policy loss: {total_policy_loss / n_updates:.4f}")
            print(f"  Value loss: {total_value_loss / n_updates:.4f}")
            print(f"  Entropy: {total_entropy / n_updates:.4f}")
            print(f"  Teacher loss: {total_teacher_loss / n_updates:.4f}")
            print(f"  Grad norm: {grad_norm:.4f}")
            print(f"  Entropy coef: {self.current_entropy_coef:.6f}")
            print(f"  Samples/update: {samples_per_update} | Total samples: {self.steps}")

        self.buffer.clear()

    # -------- Sequence helpers --------
    def _get_sequence_feats(self) -> Tuple[np.ndarray, int]:
        """Return (seq_feats[L,293], true_len) truncated to last N, optionally with BOS handled in model."""
        seq_padded, seq_len = pad_truncate_seq(
            self._history_feats293,
            self.config.max_seq_len,
            self.config.state_dim,
            start_token=self._start_token_np,
            has_start=self._history_has_start
        )
        return seq_padded, seq_len

    def _build_seq_batch(self, seq_arr: np.ndarray) -> Tuple[np.ndarray, int]:
        L = seq_arr.shape[0]
        N = self.config.max_seq_len
        if L < N:
            pad = np.zeros((N - L, self.config.state_dim), dtype=np.float32)
            seq_padded = np.concatenate([seq_arr, pad], axis=0)
        else:
            seq_padded = seq_arr[-N:]
            L = N
        return seq_padded, L

    def _append_history_feature(self, feat: np.ndarray):
        self._history_feats293.append(feat.copy())
        limit = self.config.max_seq_len - 1 if self._history_has_start else self.config.max_seq_len
        if len(self._history_feats293) > limit:
            overflow = len(self._history_feats293) - limit
            if overflow > 0:
                del self._history_feats293[:overflow]
                if self._history_has_start:
                    self._history_has_start = False

    # -------- Action selection --------
    def action(self, board_copy, dice, player, i, train=False, train_config=None):
        """Select an action given current board + dice. Maintains sequence history."""
        board_pov = _flip_board(board_copy) if player == -1 else board_copy
        possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
        nA = len(possible_moves)
        if nA == 0:
            return []

        # Moves left flag for encoder
        moves_left = 1 + int(dice[0] == dice[1]) - i
        cur_feat = self._encode_state(board_pov, moves_left, actor_flag=0.0)

        # Update history with *current state* as final token for prediction
        self._history_states29.append(board_pov.copy())
        self._append_history_feature(cur_feat)

        # Build sequence (truncate to last N)
        seq_padded, true_len = self._get_sequence_feats()

        # Candidate features (after-states)
        cand_feats = np.stack([
            self._encode_state(b_after, moves_left - 1, actor_flag=1.0) for b_after in possible_boards
        ], axis=0).astype(np.float32)

        # Cap candidates
        if nA > self.config.max_actions:
            cand_feats = cand_feats[:self.config.max_actions]
            possible_moves = possible_moves[:self.config.max_actions]
            possible_boards = possible_boards[:self.config.max_actions]
            nA = self.config.max_actions

        mask = np.ones(nA, dtype=np.float32)

        # Tensors
        seq_t = torch.as_tensor(seq_padded[None, :, :], dtype=torch.float32, device=self.device)
        seq_len_t = torch.as_tensor([min(true_len, self.config.max_seq_len)], dtype=torch.int64, device=self.device)
        cand_t = torch.as_tensor(cand_feats[None, :, :], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, value_t = self.acnet(seq_t, seq_len_t, cand_t, mask_t)
            logits = logits.squeeze(0)
            value = value_t.squeeze(0).item()

        # Sample or greedy
        if train and not self.eval_mode:
            probs = F.softmax(logits, dim=0)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                print(f"[WARNING] Invalid probabilities detected. Logits: {logits}")
                probs = torch.ones(nA, device=self.device) / nA
            a_idx = torch.multinomial(probs, 1).item()
            log_prob = torch.log(probs[a_idx] + 1e-9).item()
        else:
            a_idx = torch.argmax(logits).item()
            log_prob = 0.0

        chosen_move = possible_moves[a_idx]
        chosen_board = possible_boards[a_idx]

        # Append post-action token (agent move)
        post_feat = self._encode_state(chosen_board, moves_left - 1, actor_flag=1.0)
        self._append_history_feature(post_feat)
        self._history_states29.append(chosen_board.copy())

        # Reward signal
        terminal_reward = 1.0 if (chosen_board[27] == 15) else 0.0
        shaped_reward = self._compute_shaped_reward(board_pov, chosen_board) if train else 0.0
        total_reward = terminal_reward + shaped_reward
        done = bool(terminal_reward > 0.0)

        # Teacher label occasionally
        teacher_idx = -1
        if (train and not self.eval_mode and self._teacher_enabled() and
            random.random() < self.config.teacher_sample_rate):
            cand29 = np.stack(possible_boards[:nA], axis=0)
            teacher_idx = self.compute_teacher_index(
                board_abs=board_copy,
                dice=dice,
                player=player,
                board_pov=board_pov,
                after_states_pov=cand29,
                pmoves=possible_moves,
                mask=mask
            )

        # Buffer push when training
        if train and not self.eval_mode:
            # Pad candidates and mask to max_actions
            if nA < self.config.max_actions:
                padA = self.config.max_actions - nA
                cand_padded = np.pad(cand_feats, ((0, padA), (0, 0)), mode='constant')
                mask_padded = np.pad(mask, (0, padA), mode='constant')
            else:
                cand_padded = cand_feats
                mask_padded = mask

            self.buffer.push(
                seq=seq_padded, seq_len=int(min(true_len, self.config.max_seq_len)),
                candidate_states=cand_padded, mask=mask_padded,
                action=a_idx, log_prob=log_prob, value=value,
                reward=total_reward, done=done, teacher_idx=teacher_idx
            )
            self.steps += 1
            if self.buffer.is_ready():
                self._ppo_update()

        if player == -1:
            chosen_move = _flip_move(chosen_move)

        return chosen_move

    # -------- Episode hooks --------
    def episode_start(self):
        self._history_states29.clear()
        self._history_feats293.clear()
        self._history_has_start = True

    def end_episode(self, outcome, final_board, perspective):
        # Could log episode-level stats here if desired
        pass

    def game_over_update(self, board, reward):
        pass

    # -------- Warmstart with pubeval (optional) --------
    def warmstart_with_pubeval(self, batch_iter, epochs: int = 1, assume_second_roll=False, lr=None, grad_clip: float = 1.0):
        """
        Warm-start using pubeval labels on single-step sequences (no history).
        batch_iter should yield (S29, C29, M), analogous to previous API.
        """
        if self.teacher_type != 'pubeval' or self.teacher_module is None:
            print("Warmstart skipped: requires pubeval teacher.")
            return
        self.acnet.train()
        opt = torch.optim.Adam(self.acnet.parameters(), lr=(self.config.lr if lr is None else lr))
        ce = torch.nn.CrossEntropyLoss(reduction="mean")

        for _ in range(epochs):
            for S29, C29, M in batch_iter:
                # Teacher labels on raw 29-d boards
                labels = np.array(
                    [self._get_teacher_label_pubeval(S29[i], C29[i], M[i]) for i in range(S29.shape[0])],
                    dtype=np.int64
                )
                # Encode current state features (single-token sequence)
                if isinstance(assume_second_roll, (list, tuple, np.ndarray)):
                    sec_flags = np.asarray(assume_second_roll, dtype=bool)
                    assert sec_flags.shape[0] == S29.shape[0]
                else:
                    sec_flags = np.full((S29.shape[0],), bool(assume_second_roll))

                S_feat = np.stack(
                    [transformer_one_hot_encoding(S29[i], bool(sec_flags[i]), actor_flag=0.0) for i in range(S29.shape[0])],
                    axis=0
                ).astype(np.float32)

                # Candidate features
                B, A = C29.shape[0], C29.shape[1]
                C_feat = np.zeros((B, A, self.config.state_dim), dtype=np.float32)
                for i in range(B):
                    for a in range(A):
                        if M[i, a] != 0.0:
                            C_feat[i, a] = transformer_one_hot_encoding(C29[i, a], False, actor_flag=1.0)

                # Pack tensors
                # Build 1-token sequences (or 0 with BOS-only; we'll use 1)
                seqs = []
                lens = []
                for i in range(B):
                    seqs.append(S_feat[i:i+1, :])  # (1, 293)
                    lens.append(1)
                maxL = 1
                seq_pad = np.zeros((B, self.config.max_seq_len, self.config.state_dim), dtype=np.float32)
                for i in range(B):
                    seq_pad[i, :1, :] = seqs[i]
                lens = np.array(lens, dtype=np.int64)

                St = torch.as_tensor(seq_pad, dtype=torch.float32, device=self.device)
                Lt = torch.as_tensor(lens, dtype=torch.int64, device=self.device)
                Ct = torch.as_tensor(C_feat, dtype=torch.float32, device=self.device)
                Mt = torch.as_tensor(M, dtype=torch.float32, device=self.device)

                logits, _ = self.acnet(St, Lt, Ct, Mt)
                logits = logits.masked_fill(Mt == 0, -1e9)
                y = torch.as_tensor(labels, dtype=torch.long, device=self.device)
                loss = ce(logits, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.acnet.parameters(), grad_clip)
                opt.step()


# ------------- Module-level interface -------------
_default_agent = None
_loaded_from_disk = False
CHECKPOINT_PATH = Path("checkpoints/best_ppo_transformer.pt")


def _get_agent():
    global _default_agent
    if _default_agent is None:
        device = get_device()
        try:
            import pubeval_player as pubeval
            print("Pubeval module loaded for teacher signal")
        except ImportError:
            pubeval = None
            print("Warning: Could not import pubeval, teacher signal disabled")
        teacher_mode = 'pubeval' if pubeval is not None else 'none'
        _default_agent = PPOAgent(
            config=CFG,
            device=device,
            teacher_mode=teacher_mode,
            teacher_module=pubeval
        )
        print("Transformer PPO agent initialized with improved hyperparameters")
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
