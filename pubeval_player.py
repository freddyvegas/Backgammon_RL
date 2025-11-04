# pubeval_numba.py
# Pure-Python + Numba version of Tesauro's pubeval

import numpy as np
from numba import njit
import backgammon
import flipped_agent

# ---- weights copied from Tesauro (float32 for speed) ----
wr = np.array([
    0.00000,-0.17160,0.27010,0.29906,-0.08471,0.00000,-1.40375,-1.05121,0.07217,-0.01351,
    0.00000,-1.29506,-2.16183,0.13246,-1.03508,0.00000,-2.29847,-2.34631,0.17253,0.08302,
    0.00000,-1.27266,-2.87401,-0.07456,-0.34240,0.00000,-1.34640,-2.46556,-0.13022,-0.01591,
    0.00000,0.27448,0.60015,0.48302,0.25236,0.00000,0.39521,0.68178,0.05281,0.09266,
    0.00000,0.24855,-0.06844,-0.37646,0.05685,0.00000,0.17405,0.00430,0.74427,0.00576,
    0.00000,0.12392,0.31202,-0.91035,-0.16270,0.00000,0.01418,-0.10839,-0.02781,-0.88035,
    0.00000,1.07274,2.00366,1.16242,0.22520,0.00000,0.85631,1.06349,1.49549,0.18966,
    0.00000,0.37183,-0.50352,-0.14818,0.12039,0.00000,0.13681,0.13978,1.11245,-0.12707,
    0.00000,-0.22082,0.20178,-0.06285,-0.52728,0.00000,-0.13597,-0.19412,-0.09308,-1.26062,
    0.00000,3.05454,5.16874,1.50680,5.35000,0.00000,2.19605,3.85390,0.88296,2.30052,
    0.00000,0.92321,1.08744,-0.11696,-0.78560,0.00000,-0.09795,-0.83050,-1.09167,-4.94251,
    0.00000,-1.00316,-3.66465,-2.56906,-9.67677,0.00000,-2.77982,-7.26713,-3.40177,-12.32252,
    0.00000,3.42040
], dtype=np.float32)

wc = np.array([
     0.25696,-0.66937,-1.66135,-2.02487,-2.53398,-0.16092,-1.11725,-1.06654,-0.92830,-1.99558,
    -1.10388,-0.80802,0.09856,-0.62086,-1.27999,-0.59220,-0.73667,0.89032,-0.38933,-1.59847,
    -1.50197,-0.60966,1.56166,-0.47389,-1.80390,-0.83425,-0.97741,-1.41371,0.24500,0.10970,
    -1.36476,-1.05572,1.15420,0.11069,-0.38319,-0.74816,-0.59244,0.81116,-0.39511,0.11424,
    -0.73169,-0.56074,1.09792,0.15977,0.13786,-1.18435,-0.43363,1.06169,-0.21329,0.04798,
    -0.94373,-0.22982,1.22737,-0.13099,-0.06295,-0.75882,-0.13658,1.78389,0.30416,0.36797,
    -0.69851,0.13003,1.23070,0.40868,-0.21081,-0.64073,0.31061,1.59554,0.65718,0.25429,
    -0.80789,0.08240,1.78964,0.54304,0.41174,-1.06161,0.07851,2.01451,0.49786,0.91936,
    -0.90750,0.05941,1.83120,0.58722,1.28777,-0.83711,-0.33248,2.64983,0.52698,0.82132,
    -0.58897,-1.18223,3.35809,0.62017,0.57353,-0.07276,-0.36214,4.37655,0.45481,0.21746,
     0.10504,-0.61977,3.54001,0.04612,-0.18108,0.63211,-0.87046,2.47673,-0.48016,-1.27157,
     0.86505,-1.11342,1.24612,-0.82385,-2.77082,1.23606,-1.59529,0.10438,-1.30206,-4.11520,
     5.62596,-2.75800
], dtype=np.float32)

# allocate once (Numba needs fixed dtypes)
ZERO122 = np.zeros(122, dtype=np.float32)

@njit(cache=True)
def _setx(pos, x):
    """
    Replicates setx() from Tesauro's C code.
    pos: int32 array, length 27 or 28 (we use indices 0..26)
    x: preallocated float32 array of length 122
    """
    # zero
    for j in range(122):
        x[j] = 0.0

    # encode board locations 24..1 into features
    # Note: pos[25-j] is used in original C
    for j in range(1, 25):
        jm1 = j - 1
        n = pos[25 - j]
        base = 5 * jm1
        if n != 0:
            if n == -1: x[base + 0] = 1.0
            if n ==  1: x[base + 1] = 1.0
            if n >=  2: x[base + 2] = 1.0
            if n ==  3: x[base + 3] = 1.0
            if n >=  4: x[base + 4] = (n - 3) / 2.0

    # opponent barmen (pos[0] negative for opponent on bar)
    x[120] = -(pos[0]) / 2.0
    # computer men off (pos[26] positive)
    x[121] =  (pos[26]) / 15.0

@njit(cache=True)
def _pubeval_scalar(race, pos):
    """
    race: 0 or 1
    pos: int32 array with indexes [0..26] meaningful
    returns: float32 score
    """
    # auto-win if all borne off
    if pos[26] == 15:
        return 9.9999999e7

    x = np.empty(122, dtype=np.float32)
    _setx(pos, x)
    # pick weights
    if race == 1:
        s = 0.0
        for i in range(122):
            s += wr[i] * x[i]
        return s
    else:
        s = 0.0
        for i in range(122):
            s += wc[i] * x[i]
        return s

@njit(cache=True)
def israce(board):
    """
    Returns 1 if out of contact (pure race), else 0.
    board: numpy array length 29 (same as env) with indices 1..24 points.
    """
    # gather positions of p1 (>0) and p2 (<0)
    # In Numba, avoid np.where on booleans; loop instead
    last_p1 = -1   # highest index with >0
    first_p2 = 25  # lowest index with <0
    count_p1 = 0
    count_p2 = 0
    for i in range(1, 25):
        v = board[i]
        if v > 0:
            last_p1 = i
            count_p1 += 1
        elif v < 0:
            if i < first_p2:
                first_p2 = i
            count_p2 += 1

    if (count_p2 == 0) or (count_p1 == 0):
        return 1
    if last_p1 < first_p2:
        return 1
    return 0

def pubeval_flip(board):
    """
    Convert your environment's 29-length board into Tesauro's layout
    expected by pubeval (indexes 0..26 used).
    Mirrors your previous Python wrapper logic exactly.
    """
    b = np.copy(board)  # float by default; we cast later
    # swap bar/opponent off per your wrapper
    b[[0, 26]] = b[[26, 27]]
    b[27] = b[28]
    b = b[:-1]  # drop last to make length 28 (indices 0..27; 27 unused in C)
    # We'll only rely on indices 0..26
    return b

def _to_int32_view(b28):
    """
    Ensure we have a contiguous int32 view for Numba function.
    """
    arr = np.asarray(b28, dtype=np.int32)
    # If array is length 28, indices 0..26 are valid for pubeval
    return arr

def action(board, dice, oplayer, i=0, **_):
    """
    Pure-Python replacement for the ctypes-based pubeval agent.

    - Flip to a fixed 'computer' perspective like your original code.
    - Enumerate legal after-states via Backgammon.legal_moves.
    - Score each after-state with Numba pubeval.
    - Pick argmax and unflip move if needed.
    """
    flipped_player = -1
    if flipped_player == oplayer:
        board_eff = flipped_agent.flip_board(np.copy(board))
        player = -flipped_player
    else:
        board_eff = board
        player = oplayer

    race = int(israce(board_eff))
    possible_moves, possible_boards = backgammon.legal_moves(board_eff, dice, player)
    na = len(possible_moves)
    if na == 0:
        return []

    # score all candidate after-states
    scores = np.empty(na, dtype=np.float32)
    for j in range(na):  # <- don't shadow the keyword arg 'i'
        pb = pubeval_flip(possible_boards[j])  # map to pubeval's layout
        pos = _to_int32_view(pb)
        scores[j] = _pubeval_scalar(race, pos)

    best_idx = int(np.argmax(scores))
    chosen = possible_moves[best_idx]

    if flipped_player == oplayer:  # map move back to original view
        chosen = flipped_agent.flip_move(chosen)
    return chosen
