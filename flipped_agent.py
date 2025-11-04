#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast flipped-agent for Backgammon.
This agent always perceives itself as player +1 by flipping the board.

Optimizations:
- Precomputed index maps
- Numba JIT acceleration for flip_board / flip_move
- Compatible with Numba-accelerated Backgammon module
"""

import numpy as np
import backgammon
from numba import njit

# ============================================================
# Precomputed index map
# ============================================================
_FLIP_IDX = np.array(
    [0, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13,
     12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 25, 28, 27],
    dtype=np.int32
)

@njit(cache=True)
def _flip_board_numba(board):
    """
    Fast board flip. Returns new array view.
    board: np.ndarray length 29 (int32 or float32)
    """
    flipped = np.empty(29, dtype=board.dtype)
    for i in range(29):
        flipped[i] = -board[_FLIP_IDX[i]]
    return flipped


@njit(cache=True)
def _flip_move_numba(move):
    """
    Flips a move or a sequence of moves.
    move: (k,2) array of int32 (k=1 or 2)
    Returns flipped version (same shape).
    """
    if move.shape[0] == 0:
        return move
    flipped_move = np.empty_like(move)
    for i in range(move.shape[0]):
        flipped_move[i, 0] = _FLIP_IDX[move[i, 0]]
        flipped_move[i, 1] = _FLIP_IDX[move[i, 1]]
    return flipped_move


# ============================================================
# Public API
# ============================================================

def flip_board(board_copy: np.ndarray) -> np.ndarray:
    """Return flipped board (Numba-accelerated)."""
    board = np.asarray(board_copy, dtype=np.int32)
    return _flip_board_numba(board)


def flip_move(move: np.ndarray) -> np.ndarray:
    """Return flipped move or sequence (Numba-accelerated)."""
    if len(move) == 0:
        return move
    mv = np.asarray(move, dtype=np.int32)
    return _flip_move_numba(mv)


def action(board_copy, dice, player, i, **_):
    """
    Generic flipped agent action.
    Flips board so agent always acts as player +1.
    Currently plays random move (policy placeholder).
    """
    if player == -1:
        board_copy = flip_board(board_copy)

    # All legal moves (agent always sees itself as +1)
    possible_moves, _ = backgammon.legal_moves(board_copy, dice, player=1)
    if not possible_moves:
        return []

    # Random choice (placeholder for learned policy)
    move = possible_moves[np.random.randint(len(possible_moves))]

    # Flip move back if we flipped the board
    if player == -1:
        move = flip_move(move)
    return move
