import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import backgammon
import torch

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
