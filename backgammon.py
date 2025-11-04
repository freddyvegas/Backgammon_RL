#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Backgammon core with Numba acceleration.

Public API preserved:
- init_board() -> np.ndarray (len 29)
- roll_dice() -> np.ndarray (shape (2,))
- game_over(board) -> bool
- legal_move(board, die, player) -> List[np.ndarray([start,end])]
- legal_moves(board, dice, player) ->
      (List[np.ndarray shape (k,2)], List[np.ndarray len 29])  # k in {1,2}
- update_board(board, move, player) -> np.ndarray (len 29)

Conventions (same as original):
- Points: indices 1..24
- Bar: index 25 (player +1), 26 (player -1)
- Off/borne: index 27 (player +1, positive to 15), 28 (player -1, negative to -15)
- player ∈ {+1, -1}
"""

from typing import List, Tuple
import numpy as np
from numba import njit, types
from numba.typed import List as NList


# ========== Public helpers ==========

def init_board() -> np.ndarray:
    b = np.zeros(29, dtype=np.int32)
    # black (-1) pieces
    b[1]  = -2
    b[12] = -5
    b[17] = -3
    b[19] = -5
    # white (+1) pieces
    b[6]  = 5
    b[8]  = 3
    b[13] = 5
    b[24] = 2
    return b


def roll_dice() -> np.ndarray:
    # keep np.random (fast enough vs Python's random)
    return np.random.randint(1, 7, size=2, dtype=np.int32)


def game_over(board: np.ndarray) -> bool:
    return (board[27] == 15) or (board[28] == -15)


def pretty_print(board: np.ndarray) -> None:
    s = str(np.array2string(board[1:13]) + '\n' +
            np.array2string(board[24:12:-1]) + '\n' +
            np.array2string(board[25:29]))
    print("board:\n", s)


def check_for_error(board: np.ndarray) -> bool:
    # returns True if error
    # (sum positives) == 15 and (sum negatives) == -15
    pos_sum = int(board[board > 0].sum())
    neg_sum = int(board[board < 0].sum())
    if (pos_sum != 15) or (neg_sum != -15):
        print("Too many or too few pieces on board!")
        return True
    return False


# ========== Numba kernels (private) ==========

# NOTE: we work on int32 arrays inside kernels for speed & simplicity.
# We also represent a single "move" as a 2-int array [start, end].
# For "move sequences" used by legal_moves (at most 2 ply here),
# we use a fixed (2,2) array and fill unused rows with (-1,-1).

MOVE_PAD = np.array([-1, -1], dtype=np.int32)

@njit(cache=True)
def _has_checker(board: np.ndarray, idx: int, player: int) -> bool:
    # player=+1: board[idx] > 0, player=-1: board[idx] < 0
    return (board[idx] * player) > 0


@njit(cache=True)
def _point_is_open(board: np.ndarray, idx: int, player: int) -> bool:
    """
    A point is blocked ONLY if it has >= 2 opponent checkers.
    Otherwise it's open (empty, own checkers, or single opponent blot).
    """
    v = board[idx]
    if player == 1:
        # opponent checkers are negative; blocked if v <= -2
        return v > -2
    else:
        # opponent checkers are positive; blocked if v >= 2
        return v < 2

@njit(cache=True)
def _bearing_all_home(board: np.ndarray, player: int) -> bool:
    """
    player=+1: all checkers are on points 1..6 (no checkers 7..24)
    player=-1: all checkers are on points 19..24 (no checkers 1..18)
    """
    if player == 1:
        # any piece on 7..24?
        for i in range(7, 25):
            if board[i] > 0:
                return False
        return True
    else:
        # any piece on 1..18?
        for i in range(1, 19):
            if board[i] < 0:
                return False
        return True


@njit(cache=True)
def _rightmost_checker_1_to_6(board: np.ndarray) -> int:
    """
    For player +1 bearing off: return highest occupied pip in 1..6 (>=1), else -1.
    """
    for p in range(6, 0, -1):
        if board[p] > 0:
            return p
    return -1


@njit(cache=True)
def _leftmost_checker_19_to_24(board: np.ndarray) -> int:
    """
    For player -1 bearing off: return lowest occupied pip in 19..24 (<=24), else -1.
    """
    for p in range(19, 25):
        if board[p] < 0:
            return p
    return -1


@njit(cache=True)
def _legal_move_single(board: np.ndarray, die: int, player: int) -> NList:  # returns typed.List of (2,)
    moves = NList()
    if player == 1:
        # enter from bar?
        if board[25] > 0:
            start_pip = 25 - die
            if start_pip > 0 and _point_is_open(board, start_pip, player):
                moves.append(np.array([25, start_pip], dtype=np.int32))
            return moves  # must enter from bar if possible

        # bearing off?
        if _bearing_all_home(board, 1):
            # exact bear off?
            if board[die] > 0:
                moves.append(np.array([die, 27], dtype=np.int32))
            else:
                # if no checker can play exactly die, may bear the highest
                s = _rightmost_checker_1_to_6(board)
                if s != -1 and s < die:
                    moves.append(np.array([s, 27], dtype=np.int32))

        # normal moves
        for s in range(1, 25):
            if board[s] > 0:
                end_pip = s - die
                if end_pip > 0 and _point_is_open(board, end_pip, player):
                    moves.append(np.array([s, end_pip], dtype=np.int32))

    else:  # player == -1
        # enter from bar?
        if board[26] < 0:
            start_pip = die
            if start_pip < 25 and _point_is_open(board, start_pip, player):
                moves.append(np.array([26, start_pip], dtype=np.int32))
            return moves  # must enter from bar if possible

        # bearing off?
        if _bearing_all_home(board, -1):
            # exact bear off?
            src = 25 - die
            if src > 0 and board[src] < 0:
                moves.append(np.array([src, 28], dtype=np.int32))
            else:
                # if no checker can play exactly die, may bear lowest
                s = _leftmost_checker_19_to_24(board)
                if s != -1 and (6 - (s - 19)) < die:
                    moves.append(np.array([s, 28], dtype=np.int32))

        # normal moves
        for s in range(1, 25):
            if board[s] < 0:
                end_pip = s + die
                if end_pip < 25 and _point_is_open(board, end_pip, player):
                    moves.append(np.array([s, end_pip], dtype=np.int32))

    return moves


@njit(cache=True)
def _apply_move_inplace(board: np.ndarray, move: np.ndarray, player: int) -> None:
    """
    Mutates board with a single move [start,end] for given player.
    Handles hits to bar.
    """
    startPip = int(move[0])
    endPip   = int(move[1])
    if startPip < 0:  # padded / no-op
        return

    # if hitting opponent blot
    if endPip >= 0 and abs(board[endPip]) == 1 and (board[endPip] * (-player)) > 0:
        # move opponent checker to its bar
        board[endPip] = 0
        jail = 25 + (1 if player == 1 else 0)  # 26 for player -1, 25 for player +1?
        # In original: jail = 25 + (player==1)  -> for hit by +1, opponent goes to index 26 (correct)
        # That expression yields 26 when player==1 else 25.
        board[jail] -= player  # subtract player's sign pushes opponent count with sign

    # move checker
    board[startPip] -= player
    board[endPip]   += player


@njit(cache=True)
def _update_board(board: np.ndarray, move: np.ndarray, player: int) -> np.ndarray:
    """
    Accepts either:
      - a single move shape (2,)
      - a sequence shape (k,2) with k=1 or 2 (second row may be padding -1,-1)
    """
    nb = board.copy()

    if move.ndim == 1:
        # single [start, end]
        _apply_move_inplace(nb, move, player)
        return nb

    # (k,2) sequence; apply first row
    _apply_move_inplace(nb, move[0], player)

    # apply second row only if present and not padded
    if move.shape[0] > 1:
        if move[1, 0] >= 0:  # not (-1, -1)
            _apply_move_inplace(nb, move[1], player)

    return nb


@njit(cache=True)
def _pack_seq(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    out = np.empty((2, 2), dtype=np.int32)
    out[0, 0] = m1[0]; out[0, 1] = m1[1]
    if m2.shape[0] == 2:
        out[1, 0] = m2[0]; out[1, 1] = m2[1]
    else:
        out[1, 0] = -1; out[1, 1] = -1
    return out


@njit(cache=True)
def _legal_moves_impl(board: np.ndarray, d1: int, d2: int, player: int):
    """
    Returns (typed list of (2,2) int32 move sequences, typed list of board int32)
    Each sequence row is a single move [start,end]; second row (-1,-1) means no second move.
    """
    seqs = NList()
    boards = NList()

    # first order: d1 then d2
    m1s = _legal_move_single(board, d1, player)
    for m1 in m1s:
        tb = _update_board(board, m1, player)
        m2s = _legal_move_single(tb, d2, player)
        for m2 in m2s:
            seq = _pack_seq(m1, m2)
            seqs.append(seq)
            boards.append(_update_board(board, seq, player))

    # if non-doubles, also try d2 then d1
    if d1 != d2:
        m1s = _legal_move_single(board, d2, player)
        for m1 in m1s:
            tb = _update_board(board, m1, player)
            m2s = _legal_move_single(tb, d1, player)
            for m2 in m2s:
                seq = _pack_seq(m1, m2)
                seqs.append(seq)
                boards.append(_update_board(board, seq, player))

    # if no pairs, allow single moves
    if len(seqs) == 0:
        # singles for d1
        m1s = _legal_move_single(board, d1, player)
        for m in m1s:
            seq = _pack_seq(m, MOVE_PAD)
            seqs.append(seq)
            boards.append(_update_board(board, seq, player))
        # singles for d2 (if not doubles)
        if d1 != d2:
            m1s = _legal_move_single(board, d2, player)
            for m in m1s:
                seq = _pack_seq(m, MOVE_PAD)
                seqs.append(seq)
                boards.append(_update_board(board, seq, player))

    return seqs, boards


# ========== Public API (Numba-backed) ==========

def legal_move(board: np.ndarray, die: int, player: int) -> List[np.ndarray]:
    """
    For compatibility with your original code: returns Python list of [start,end] arrays.
    """
    b = np.asarray(board, dtype=np.int32)
    seqs = _legal_move_single(b, int(die), int(player))
    # Convert typed.List -> Python list
    return [np.array([m[0], m[1]], dtype=np.int32) for m in seqs]


def legal_moves(board: np.ndarray, dice: np.ndarray, player: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      moves:  list of np.ndarray shape (k,2) with k in {1,2} (sequence of 1 or 2 single-moves)
      boards: list of np.ndarray shape (29,) with resulting after-state boards
    """
    b = np.asarray(board, dtype=np.int32)
    d1 = int(dice[0]); d2 = int(dice[1])
    seqs, boards = _legal_moves_impl(b, d1, d2, int(player))

    moves_py: List[np.ndarray] = []
    boards_py: List[np.ndarray] = []

    for seq, bb in zip(seqs, boards):
        # trim second row if padded
        if seq[1, 0] < 0:
            mv = seq[0:1, :].copy()  # shape (1,2)
        else:
            mv = seq.copy()          # shape (2,2)
        moves_py.append(mv)
        # cast back to int32 (already) and pad to len 29 (already)
        # ensure numpy array (Numba array is already numpy)
        boards_py.append(bb.copy())

    return moves_py, boards_py


def update_board(board: np.ndarray, move: np.ndarray, player: int) -> np.ndarray:
    """
    Accepts either a single move [start,end] or a sequence shape (k,2).
    """
    b = np.asarray(board, dtype=np.int32)
    mv = np.asarray(move, dtype=np.int32)
    # normalize to (2,2) with padding
    if mv.ndim == 1:
        seq = _pack_seq(mv, MOVE_PAD)
    elif mv.shape == (1, 2):
        seq = _pack_seq(mv[0], MOVE_PAD)
    elif mv.shape == (2, 2):
        seq = mv
    else:
        # fallback: treat as single
        seq = _pack_seq(mv.reshape(2), MOVE_PAD)
    return _update_board(b, seq, int(player))

