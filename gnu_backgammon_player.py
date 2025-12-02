#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent that defers move selection to the official GNU Backgammon engine.

The gnubg Python package bundles the same neural networks that power the
desktop client, so we simply convert our environment's board representation
into the TanBoard format expected by gnubg.best_move(), query the engine, and
convert its answer back to our move format.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

import backgammon
import flipped_agent

try:
    import gnubg  # type: ignore
except ImportError as exc:  # pragma: no cover - easier to debug upfront
    raise RuntimeError(
        "The 'gnubg' package is required for gnu_backgammon_player. "
        "Install it via `python3 -m pip install gnubg`."
    ) from exc


TanBoard = List[List[int]]  # alias that matches gnubg's expectations


def _board_to_tan(board: np.ndarray, player: int) -> TanBoard:
    """
    Convert our 29-length board (absolute orientation) into gnubg's TanBoard
    where anBoard[1] holds the acting player's checkers (from their POV) and
    anBoard[0] holds the opponent's mirrored layout.
    """
    if player not in (1, -1):
        raise ValueError(f"player must be ±1, got {player}")

    arr = np.array(board, copy=True, dtype=np.int32)
    if arr.shape[0] != 29:
        raise ValueError(f"board must have length 29, got {arr.shape[0]}")

    if player == -1:
        arr = flipped_agent.flip_board(arr)

    player_board = [0] * 25
    opponent_board = [0] * 25

    for pip in range(1, 25):
        val = int(arr[pip])
        if val > 0:
            player_board[pip - 1] = val
        elif val < 0:
            # Opponent points are mirrored relative to the acting player's POV.
            opponent_board[24 - pip] = -val

    # Index 24 stores the number of checkers on the bar for that player.
    player_board[24] = int(max(arr[25], 0))
    opponent_board[24] = int(max(-arr[26], 0))

    return [opponent_board, player_board]


def _normalize_gnubg_move(result: Sequence) -> List[Tuple[int, int]]:
    """
    Extract the contiguous prefix of (start, end) tuples from gnubg.best_move().
    gnubg can optionally append extra metadata, so we only keep the move pairs.
    """
    moves: List[Tuple[int, int]] = []
    for entry in result:
        if isinstance(entry, (str, bytes)):
            break
        if not isinstance(entry, Iterable):
            break
        try:
            if len(entry) != 2:  # type: ignore[arg-type]
                break
            start, end = entry  # type: ignore[assignment]
            moves.append((int(start), int(end)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            break
    return moves


def _tan_move_to_env(move_pairs: List[Tuple[int, int]], player: int) -> np.ndarray:
    """
    Convert gnubg's move pairs back into our absolute board orientation.

    - gnubg returns up to four steps for doubles; we only execute at most two
      per env action, so we keep the leading pairs.
    - gnubg encodes bearing off as destination 0, whereas our env uses 27/28.
    """
    if not move_pairs:
        return np.empty((0, 2), dtype=np.int32)

    trimmed = move_pairs[:2]  # environment plays two dice per action
    arr = np.array(trimmed, dtype=np.int32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)

    # Map gnubg's off-board index (0) to our borne-off slot (27).
    # Player -1 will be flipped below which turns 27 into 28.
    off_mask = arr[:, 1] == 0
    if np.any(off_mask):
        arr = arr.copy()
        arr[off_mask, 1] = 27

    if player == -1:
        arr = flipped_agent.flip_move(arr)
    return arr


def action(
    board: np.ndarray,
    dice: Sequence[int],
    player: int,
    i: int = 0,
    gnubg_kwargs: dict | None = None,
    **_,
) -> List[List[int]] | np.ndarray | List[int]:
    """
    Select the best move according to GNU Backgammon.

    Args:
        board: np.ndarray shape (29,) following the project convention.
        dice: sequence of two ints for the current roll.
        player: +1 or -1.
        i: unused (maintained for compatibility).
        gnubg_kwargs: optional dict passed verbatim to gnubg.best_move.

    Returns:
        [] when no legal move exists, otherwise an array shaped (k, 2)
        with (start, end) pip numbers that our environment can consume.
    """
    dice_vals = np.asarray(dice, dtype=np.int32)
    legal_moves, _ = backgammon.legal_moves(board, dice_vals, player)
    if not legal_moves:
        return []

    tan_board = _board_to_tan(board, player)

    kwargs = gnubg_kwargs or {}
    try:
        result = gnubg.best_move(
            tan_board,
            int(dice_vals[0]),
            int(dice_vals[1]),
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover - engine failures should be rare
        raise RuntimeError(f"gnubg.best_move failed: {exc}") from exc

    move_pairs = _normalize_gnubg_move(result if isinstance(result, Sequence) else [])
    move = _tan_move_to_env(move_pairs, player)

    if move.size == 0:
        return []

    move_shape = move.shape
    for candidate in legal_moves:
        cand_arr = np.asarray(candidate)
        if cand_arr.shape == move_shape and np.array_equal(cand_arr, move):
            return move

    # Fallback: gnubg occasionally returns equivalent but differently ordered
    # sequences; in that unlikely case we simply default to the first legal move.
    return legal_moves[0]
