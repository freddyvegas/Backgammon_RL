#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast random agent for Backgammon.

- Keeps the same public API: action(board_copy, dice, player, i, **kwargs)
- Uses NumPy RNG and avoids extra Python overhead.
- Relies on your Numba-accelerated Backgammon.legal_moves for speed.
"""

import numpy as np
import backgammon  # your optimized version

# Optional: independent RNG stream (reproducible if you seed it)
_rng = np.random.default_rng()

def action(board_copy, dice, player, i, **_):
    """
    Choose a uniformly random legal move sequence.
    Returns:
      []          if no legal move exists
      (k,2) array if a legal sequence exists (k in {1,2} per your API)
    """
    # Get all legal after-states for this roll
    possible_moves, _ = backgammon.legal_moves(board_copy, dice, player)

    # No legal moves (pass)
    if not possible_moves:
        return []

    # Single fast path: if only one legal sequence, return it
    if len(possible_moves) == 1:
        return possible_moves[0]

    # Uniform random choice among sequences
    idx = _rng.integers(len(possible_moves))
    return possible_moves[idx]

