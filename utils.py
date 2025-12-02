import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import backgammon
import torch


STATE_DIM = 24 * 2 * 6 + 4 + 1  # 293
TRANSFORMER_EXTRA_FEATURES = 2   # actor flag + start indicator
TRANSFORMER_STATE_DIM = STATE_DIM + TRANSFORMER_EXTRA_FEATURES

def get_device():
    """
    Automatically detect and return the best available device.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Available but use with caution for PPO
    else:
        return "cpu"

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

# -------------------- Features --------------------

def one_hot_encoding(board29, nSecondRoll: bool):
    """
    board29: float/int array length 29 in +1 POV:
      indices 1..24 = points (checkers for +1 positive, for -1 negative)
      25,26,27,28    = +bar, -bar, +off, -off   (same indexing you use)
      index 0 is ignored here (kept for compatibility with your 29-vector)
    nSecondRoll: True on the first move of doubles, else False
    returns: np.float32 shape (nx,)
    nx = 24 * 2 * 6 + 4 + 1  # = 293
    """
    # Fast path: handle torch tensors via the torch implementation
    if torch.is_tensor(board29):
        # Use torch encoder and round-trip to NumPy for legacy code
        return one_hot_encoding_torch(board29, nSecondRoll).cpu().numpy()

    oneHot = np.zeros(24 * 2 * 6 + 4 + 1, dtype=np.float32)

    # +1 side bins
    for i in range(0, 5):
        idx = np.where(board29[1:25] == i)[0] - 1
        if idx.size > 0:
            oneHot[i*24 + idx] = 1
    idx = np.where(board29[1:25] >= 5)[0] - 1
    if idx.size > 0:
        oneHot[5*24 + idx] = 1
    # -1 side bins
    for i in range(0, 5):
        idx = np.where(board29[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6*24 + i*24 + idx] = 1
    idx = np.where(board29[1:25] <= -5)[0] - 1
    if idx.size > 0:
        oneHot[6*24 + 5*24 + idx] = 1
    # bars/offs + second-roll flag
    oneHot[12 * 24 + 0] = board29[25]
    oneHot[12 * 24 + 1] = board29[26]
    oneHot[12 * 24 + 2] = board29[27]
    oneHot[12 * 24 + 3] = board29[28]
    oneHot[12 * 24 + 4] = 1.0 if nSecondRoll else 0.0

    return oneHot

def transformer_one_hot_encoding(board29, nSecondRoll: bool, actor_flag: float):
    """One-hot features with actor flag and start indicator slot."""
    base = one_hot_encoding(board29, nSecondRoll)
    token = np.zeros(TRANSFORMER_STATE_DIM, dtype=np.float32)
    token[:STATE_DIM] = base
    token[-2] = 1.0 if actor_flag > 0 else 0.0  # actor channel
    token[-1] = 0.0  # start indicator (set elsewhere)
    return token

def one_hot_encoding_torch(board29_t, nSecondRoll):
    if not torch.is_tensor(board29_t):
        board29_t = torch.as_tensor(board29_t, dtype=torch.float32)
    else:
        board29_t = board29_t.to(dtype=torch.float32)
    device = board29_t.device

    pts = board29_t[..., 1:25]  # (..., 24)
    out_shape = board29_t.shape[:-1] + (STATE_DIM,)
    out = torch.zeros(out_shape, device=device, dtype=torch.float32)

    # +1 side
    for i in range(5):
        mask = (pts == float(i)).to(torch.float32)
        out[..., i * 24:(i + 1) * 24] = mask
    mask_ge5_pos = (pts >= 5).to(torch.float32)
    out[..., 5 * 24:6 * 24] = mask_ge5_pos

    # -1 side
    neg_pts = -pts
    base = 6 * 24
    for i in range(5):
        mask = (neg_pts == float(i)).to(torch.float32)
        out[..., base + i * 24:base + (i + 1) * 24] = mask
    mask_ge5_neg = (neg_pts >= 5).to(torch.float32)
    out[..., base + 5 * 24:base + 6 * 24] = mask_ge5_neg

    # bars/offs
    out[..., 12 * 24 + 0] = board29_t[..., 25]
    out[..., 12 * 24 + 1] = board29_t[..., 26]
    out[..., 12 * 24 + 2] = board29_t[..., 27]
    out[..., 12 * 24 + 3] = board29_t[..., 28]

    # second-roll flag
    flag = torch.as_tensor(float(nSecondRoll), device=device, dtype=torch.float32)
    if flag.shape != out.shape[:-1]:
        flag = flag.expand(out.shape[:-1])
    out[..., 12 * 24 + 4] = flag

    return out

def transformer_one_hot_encoding_torch(board29_t, nSecondRoll, actor_flag):
    """Torch variant of transformer token encoding."""
    base = one_hot_encoding_torch(board29_t, nSecondRoll)
    if torch.is_tensor(actor_flag):
        actor_val = actor_flag.to(dtype=torch.float32, device=base.device)
    else:
        actor_val = torch.as_tensor(float(actor_flag > 0), dtype=torch.float32, device=base.device)
    extra_shape = base.shape[:-1] + (1,)
    actor_tensor = actor_val.expand(extra_shape)
    start_tensor = torch.zeros_like(actor_tensor)
    return torch.cat([base, actor_tensor, start_tensor], dim=-1)

def transformer_start_token():
    """Return a numpy start-of-game token."""
    token = np.zeros(TRANSFORMER_STATE_DIM, dtype=np.float32)
    token[-2] = 0.0
    token[-1] = 1.0
    return token


# --- Checkpoint helpers ---
def _ensure_dir(path: Path) -> bool:
    """Verify directory is writable, create if needed."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        testfile = path / ".write_test"
        with open(testfile, "w") as f:
            f.write("ok")
        testfile.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[Checkpoint WARNING] Cannot write to {path}: {e}")
        return False

def _safe_save_agent(agent_module, path: Path, label: str = ""):
    """Save agent with verification."""
    agent_module.save(str(path))
    if not path.exists():
        print(f"[Checkpoint WARNING] Expected to save {label or path.name} at {path}, "
              f"but no file was found.")
    else:
        print(f"[Checkpoint] Saved {label or path.name} -> {path}")

def plot_perf(perf_data, start, end, n_epochs, title="Training progress", timestamp=""):
    """Plot performance metrics."""
    if not perf_data or not any(perf_data.values()):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, data in perf_data.items():
        if data:
            xs = np.arange(start, end+1, n_epochs)
            ax.plot(xs, data, marker='o', label=label, linewidth=2)

    ax.set_xlabel("Games played", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_name = f"training_plot_{timestamp}.png"
    plt.savefig(plot_name)
    print(f"Training plot saved to {plot_name}")

def plot_perf_multi(perf_data, start_game, n_games, n_epochs, title='Training Performance', timestamp=None):
    """
    Plot performance against multiple opponents (pubeval, random, GNU BG).
    """
    import matplotlib.pyplot as plt
    
    x_vals = list(range(start_game, n_games + 1, n_epochs))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot vs Pubeval (primary baseline)
    if 'vs_baseline' in perf_data and len(perf_data['vs_baseline']) > 0:
        ax.plot(x_vals[:len(perf_data['vs_baseline'])], 
                perf_data['vs_baseline'], 
                'o-', label='vs Pubeval', linewidth=2, markersize=6)
    
    # Plot vs Random
    if 'vs_random' in perf_data and len(perf_data['vs_random']) > 0:
        ax.plot(x_vals[:len(perf_data['vs_random'])], 
                perf_data['vs_random'], 
                's-', label='vs Random', linewidth=2, markersize=6)
    
    # Plot vs GNU Backgammon
    if 'vs_gnubg' in perf_data and len(perf_data['vs_gnubg']) > 0:
        ax.plot(x_vals[:len(perf_data['vs_gnubg'])], 
                perf_data['vs_gnubg'], 
                '^-', label='vs GNU BG', linewidth=2, markersize=6)
    
    ax.set_xlabel('Games played', fontsize=12)
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Save with timestamp
    if timestamp:
        filename = f'training_plot_{timestamp}.png'
    else:
        from datetime import datetime
        filename = f'training_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to {filename}")
    plt.close()

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

def flip_to_pov_plus1(board: np.ndarray, player: int) -> np.ndarray:
    """Return a +1-POV board for encoding; do NOT use for rules."""
    if player == 1:
        return board
    b = board
    out = np.zeros_like(b)
    # points 1..24 reversed and sign-flipped so current player is positive
    out[1:25] = -b[1:25][::-1]
    # swap bars and borne-off bins, flip signs
    out[25] = -b[26]   # +1 bar gets what was -1's bar
    out[26] = -b[25]
    out[27] = -b[28]   # +1 borne-off gets what was -1's borne-off
    out[28] = -b[27]
    return out

def append_token(histories293, hist_lens, idx, board29, nSecondRoll_flag, one_hot_encoding_fn,
                 max_seq_len=None, start_flags=None):
    """
    Append a new token to the per-environment history with optional truncation.

    Args:
        histories293: list[list[np.ndarray]]
        hist_lens: list[int]
        idx: int
        board29: np.ndarray (29,)
        nSecondRoll_flag: bool
        one_hot_encoding_fn: callable returning feature vector
        max_seq_len: optional cap on history length
        start_flags: optional list[bool]; when True reserves one slot for start token
    Returns:
        True if the oldest tokens were removed due to truncation.
    """
    token = one_hot_encoding_fn(board29.astype(np.float32), nSecondRoll_flag)
    if hist_lens[idx] == 0 or not np.array_equal(histories293[idx][-1], token):
        histories293[idx].append(token)
        hist_lens[idx] += 1

    if max_seq_len is None:
        return False

    removed = False
    limit = max_seq_len
    if start_flags is not None and start_flags[idx]:
        limit = max(1, max_seq_len - 1)

    while hist_lens[idx] > limit:
        del histories293[idx][0]
        hist_lens[idx] -= 1
        removed = True
        if start_flags is not None and start_flags[idx]:
            start_flags[idx] = False
            limit = max_seq_len

    return removed

def append_token_torch(histories293, hist_lens, idx, board29, nSecondRoll_flag, device=None, max_seq_len=None):
    """Append a tokenized observation to an env history with optional truncation."""
    board29_t = torch.as_tensor(board29, dtype=torch.float32, device=device)
    token = one_hot_encoding_torch(board29_t, nSecondRoll_flag)  # (293,)

    if hist_lens[idx] == 0 or not torch.equal(histories293[idx][-1], token):
        histories293[idx].append(token)
        hist_lens[idx] += 1
        if max_seq_len is not None and hist_lens[idx] > max_seq_len:
            overflow = hist_lens[idx] - max_seq_len
            if overflow > 0:
                del histories293[idx][:overflow]
                hist_lens[idx] = max_seq_len


def pad_truncate_seq(seq_list, max_seq_len, state_dim, start_token=None, has_start=False):
    """
    Pad or truncate a list of tokens (optionally prefixed with a learnable start token).

    Args:
        seq_list: list[np.ndarray] — tokenized state history.
        max_seq_len: int — maximum sequence length.
        state_dim: int — feature dimension per token.
        start_token: optional np.ndarray(state_dim,) representing start sentinel.
        has_start: bool — whether the episode still includes the start token.
    """
    tokens = seq_list
    if has_start and start_token is not None:
        tokens = [start_token] + tokens
    L = len(tokens)
    take = min(L, max_seq_len)
    seq_padded = np.zeros((max_seq_len, state_dim), dtype=np.float32)
    if take > 0:
        seq_slice = np.stack(tokens[L - take:L], axis=0).astype(np.float32)
        seq_padded[:take, :] = seq_slice
    else:
        take = 1
    return seq_padded, take

def pad_truncate_seq_torch(history_tokens, N, D):
    """
    history_tokens: list[torch.Tensor(293,)]
    Returns: (seq_padded_np[N, D], seq_len)
    """
    import numpy as np

    L = len(history_tokens)
    seq_len = min(L, N)

    seq = torch.zeros((N, D), dtype=torch.float32)
    if seq_len > 0:
        recent = torch.stack(history_tokens[-seq_len:], dim=0).to(device=seq.device, dtype=torch.float32)
        seq[:seq_len, :] = recent
        seq_len_out = seq_len
    else:
        seq_len_out = 1  # keep at least one timestep for downstream math

    return seq.cpu().numpy(), seq_len_out


def build_histories_batch(histories293, hist_lens, max_seq_len=None, state_dim=STATE_DIM):
    """
    Build a padded batch (B, L_max, D) and corresponding length vector (B,)
    from per-env histories. Used before batch_score calls.

    Args:
        histories293: list[list[np.ndarray(293,)]]
        hist_lens: list[int]
        max_seq_len: optional int to clamp/truncate histories
        state_dim: feature dimension (defaults to STATE_DIM)

    Returns:
        hist_pad: np.ndarray (B, L_max, state_dim)
        hist_len: np.ndarray (B,)
    """
    B = len(histories293)
    if B == 0:
        L_max = max_seq_len if max_seq_len is not None else 1
        return np.zeros((0, L_max, state_dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    D = state_dim
    if max_seq_len is not None:
        L_max = max(1, max_seq_len)
    else:
        L_max = max(1, max(hist_lens))

    hist_pad = np.zeros((B, L_max, D), dtype=np.float32)
    hist_len = np.zeros(B, dtype=np.int64)

    for i in range(B):
        L_i = hist_lens[i]
        if max_seq_len is not None and L_i > max_seq_len:
            L_i = max_seq_len
        if L_i > 0:
            seq_i = np.stack(histories293[i][-L_i:], axis=0).astype(np.float32)
            hist_pad[i, :L_i, :] = seq_i
        hist_len[i] = max(1, L_i)

    return hist_pad, hist_len

def build_histories_batch_torch(h_batch, h_lens, device=None):
    B = len(h_batch)
    if B == 0:
        return torch.zeros(0, 1, STATE_DIM), torch.zeros(0, dtype=torch.long)

    L_max = max(1, max(h_lens))
    D = STATE_DIM
    if device is None:
        device = h_batch[0][0].device

    hist_pad = torch.zeros((B, L_max, D), dtype=torch.float32, device=device)
    hist_len = torch.as_tensor(h_lens, dtype=torch.long, device=device)

    for i in range(B):
        L_i = h_lens[i]
        if L_i > 0:
            seq_i = torch.stack(h_batch[i], dim=0).to(device=device, dtype=torch.float32)
            hist_pad[i, :L_i, :] = seq_i

    return hist_pad, hist_len
