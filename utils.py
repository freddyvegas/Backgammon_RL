import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import backgammon
import torch


STATE_DIM = 24 * 2 * 6 + 4 + 1  # 293

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

def append_token(histories293, hist_lens, idx, board29, nSecondRoll_flag, one_hot_encoding_fn):
    """
    Append a new 293-dim token to the per-environment history.

    Args:
        histories293: list[list[np.ndarray(293,)]]
            Global per-env list of token sequences.
        hist_lens: list[int]
            Parallel list of current sequence lengths.
        idx: int
            Environment index to update.
        board29: np.ndarray (29,)
            Current +1 POV board representation.
        nSecondRoll_flag: bool
            Whether this token should include the 'second roll' context bit.
        one_hot_encoding_fn: callable(board29, nSecondRoll_flag) -> np.ndarray(293,)
    """
    token = one_hot_encoding_fn(board29.astype(np.float32), nSecondRoll_flag)
    if hist_lens[idx] == 0 or not np.array_equal(histories293[idx][-1], token):
        histories293[idx].append(token)
        hist_lens[idx] += 1

def append_token_torch(histories293, hist_lens, idx, board29, nSecondRoll_flag, device=None):
    board29_t = torch.as_tensor(board29, dtype=torch.float32, device=device)
    token = one_hot_encoding_torch(board29_t, nSecondRoll_flag)  # (293,)

    if hist_lens[idx] == 0 or not torch.equal(histories293[idx][-1], token):
        histories293[idx].append(token)
        hist_lens[idx] += 1


def pad_truncate_seq(seq_list, max_seq_len, state_dim):
    """
    Pad or truncate a list of 293-d tokens into a (max_seq_len, state_dim) array.

    Args:
        seq_list: list[np.ndarray(293,)] — tokenized state history.
        max_seq_len: int — maximum sequence length.
        state_dim: int — feature dimension per token (usually 293).

    Returns:
        seq_padded: np.ndarray (max_seq_len, state_dim)
        seq_len: int — number of valid tokens (after truncation).
    """
    L = len(seq_list)
    take = min(L, max_seq_len)
    seq_padded = np.zeros((max_seq_len, state_dim), dtype=np.float32)
    if take > 0:
        seq_slice = np.stack(seq_list[L - take:L], axis=0).astype(np.float32)
        seq_padded[:take, :] = seq_slice
    else:
        seq_padded[0, :] = np.zeros((state_dim,), dtype=np.float32)
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


def build_histories_batch(histories293, hist_lens):
    """
    Build a padded batch (B, L_max, D) and corresponding length vector (B,)
    from per-env histories. Used before batch_score calls.

    Args:
        histories293: list[list[np.ndarray(293,)]]
        hist_lens: list[int]

    Returns:
        hist_pad: np.ndarray (B, L_max, 293)
        hist_len: np.ndarray (B,)
    """
    B = len(histories293)
    D = histories293[0][0].shape[-1] if histories293[0] else 293
    L_max = max(1, max(hist_lens))
    hist_pad = np.zeros((B, L_max, D), dtype=np.float32)
    for i in range(B):
        L_i = hist_lens[i]
        if L_i > 0:
            seq_i = np.stack(histories293[i], axis=0).astype(np.float32)
            hist_pad[i, :L_i, :] = seq_i
    return hist_pad, np.asarray(hist_lens, dtype=np.int64)

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
