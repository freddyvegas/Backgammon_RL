#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Training Script with MPS Support - FIXED VERSION

Fixes applied from code review:
1. Fixed torch.gather dtype issue (must be LongTensor)
2. Added negative terminal reward for opponent wins
3. Fixed doubles handling to match evaluation behavior
4. Fixed device selection consistency
5. Removed best_wr tracking (was never updated)
6. Changed plt.show() to plt.savefig()
7. Removed double masking of logits

NEW: Support for small/medium/large model sizes
- Small: ~80K params, 5-10x faster, great for CPU testing
- Medium: ~400K params, 2-3x faster, good for T4 GPU or M1/M2 Mac
- Large: ~1.6M params, best performance, needs good GPU

Device support: CUDA > MPS > CPU (auto-detected)
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from pathlib import Path
import os
import random
import torch
import torch.nn.functional as F
import argparse
import importlib

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
import ppo_agent as agent
from ppo_agent import one_hot_encoding
from opponent_pool import OpponentPool

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Local checkpoint directory (no Google Drive)
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 32  # or 16/32 for more parallel games

print(f"Batch self-play games: {BATCH_SIZE} games in parallel")

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

def plot_perf(perf_data, title="Training progress"):
    """Plot performance metrics."""
    if not perf_data or not any(perf_data.values()):
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, data in perf_data.items():
        if data:
            xs = np.arange(len(data))
            ax.plot(xs, data, marker='o', label=label, linewidth=2)

    ax.set_xlabel("Evaluation Checkpoint", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # FIX #11: Save instead of show (doesn't block training)
    plt.savefig("training_plot.png")
    print(f"Training plot saved to training_plot.png")

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

def play_games_batched(agent_obj, opponent, batch_size=8, training=True):
    """
    Run up to `batch_size` games in parallel and feed batched states to the network.
    Uses the same reward shaping and rollout buffer logic the agent expects.

    FIXES APPLIED:
    - FIX #2: torch.gather now uses dtype=torch.long
    - FIX #3: Added negative terminal reward when opponent wins (via new transition)
    - FIX #4: Fixed doubles handling to match evaluation (passes_left counter)
    - FIX #10: Removed double masking (ACNet.forward already masks)
    """
    finished = 0
    # Keep trajectories per environment so GAE never mixes them
    per_env_rollouts = [[] for _ in range(batch_size)]

    # Each slot holds the current env state or None if finished
    env_active = [True] * batch_size
    boards     = []
    players    = []
    dices      = []
    passes_left = []  # FIX #4: Track passes remaining per environment

    # Initialize games
    for _ in range(batch_size):
        board = backgammon.init_board()
        player = 1  # we'll always feed +1 POV into the net
        dice = backgammon.roll_dice()
        boards.append(board)
        players.append(player)
        dices.append(dice)
        # FIX #4: Initialize passes based on whether first roll is doubles
        passes_left.append(2 if dice[0] == dice[1] else 1)

    # Main loop until all batched envs finish
    while any(env_active):
        # Separate agent moves from opponent moves
        agent_envs = []  # Environments where agent (player +1) needs to act
        opponent_envs = []  # Environments where opponent (player -1) needs to act

        for idx in range(batch_size):
            if not env_active[idx]:
                continue

            player = players[idx]
            dice = dices[idx]
            board = boards[idx]

            # Check for legal moves
            pmoves, pboards = backgammon.legal_moves(board, dice, player)
            if len(pmoves) == 0:
                # No legal moves, decrement passes and switch if needed
                passes_left[idx] -= 1
                if passes_left[idx] <= 0:
                    # FIX #4: Switch players only when passes exhausted
                    players[idx] = -player
                    dices[idx] = backgammon.roll_dice()
                    passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1
                continue

            if player == 1:
                agent_envs.append((idx, dice, board, pmoves, pboards))
            else:
                opponent_envs.append((idx, dice, board, pmoves, pboards))

        # ---- Process agent moves in batch ----
        if agent_envs:
            batch_states = []
            batch_cand_states = []
            batch_masks = []
            per_env_candidates = []

            for (idx, dice, board, pmoves, pboards) in agent_envs:
                # Encode current state S in +1 POV and append roll-context bit
                board_pov = flip_to_pov_plus1(board, 1)  # player=1
                moves_left = passes_left[idx]            # 1 (normal) or 2 (first move of doubles)
                S29 = board_pov.astype(np.float32)
                nSecondRoll_now = (moves_left > 1)
                S_feat = one_hot_encoding(S29, nSecondRoll_now)   # shape (nx,)

                # Build candidates (now state_dim wide)
                cand_feats = np.zeros(
                    (agent_obj.config.max_actions, agent_obj.config.state_dim),
                    dtype=np.float32
                )

                mask = np.zeros(agent_obj.config.max_actions, dtype=np.float32)

                nA = min(len(pboards), agent_obj.config.max_actions)
                for a in range(nA):
                    after29 = flip_to_pov_plus1(pboards[a], 1)
                    nSecondRoll_next = (passes_left[idx] - 1) > 1
                    cand_feats[a]  = one_hot_encoding(after29, nSecondRoll_next)
                    mask[a] = 1.0

                batch_states.append(S_feat)
                batch_cand_states.append(cand_feats)
                batch_masks.append(mask)
                per_env_candidates.append((idx, pmoves, pboards))

            # Batched forward pass for agent
            states_np = np.stack(batch_states, axis=0)
            cand_states_np = np.stack(batch_cand_states, axis=0)
            masks_np = np.stack(batch_masks, axis=0)

            logits, values = agent_obj.batch_score(states_np, cand_states_np, masks_np)

            # FIX #10: Removed double masking - ACNet.forward already applies mask
            # Original: logits = logits.masked_fill(torch.as_tensor(masks_np, device=logits.device) == 0, -1e9)

            # Action selection
            if training and not agent_obj.eval_mode:
                probs = torch.softmax(logits, dim=-1)
                a_idxs = torch.multinomial(probs, num_samples=1).squeeze(1)

                # FIX #2: torch.gather requires LongTensor indices
                a_idxs_long = a_idxs.long()  # Ensure dtype is torch.long
                log_probs = torch.log(
                    torch.gather(probs, 1, a_idxs_long.unsqueeze(1)).squeeze(1) + 1e-9
                ).tolist()
                a_idxs = a_idxs.tolist()
            else:
                a_idxs = torch.argmax(logits, dim=-1).tolist()
                log_probs = [0.0] * len(a_idxs)

            # Apply agent actions
            for row, (idx, pmoves, pboards) in enumerate(per_env_candidates):
                a_idx = int(a_idxs[row])
                if a_idx >= len(pmoves):
                    a_idx = len(pmoves) - 1

                chosen_move = pmoves[a_idx]
                old_board = boards[idx].copy()
                boards[idx] = backgammon.update_board(boards[idx], chosen_move, 1)  # player=1

                # Compute reward
                reward = 0.0
                terminal_reward = 0.0
                if boards[idx][27] == 15:  # Agent wins
                    terminal_reward = 1.0
                elif boards[idx][28] == -15:  # Agent loses
                    terminal_reward = -1.0

                # Shaped reward
                shaped_reward = 0.0
                if training and not agent_obj.eval_mode and agent_obj.config.use_reward_shaping:
                    board_pov_old = flip_to_pov_plus1(old_board, 1)
                    board_pov_new = flip_to_pov_plus1(boards[idx], 1)
                    shaped_reward = agent_obj._compute_shaped_reward(board_pov_old, board_pov_new)

                reward = terminal_reward + shaped_reward

                # Apply reward scaling (prevents value function collapse with sparse rewards)
                reward = reward * agent_obj.config.reward_scale

                # Store in rollout buffer
                if training and not agent_obj.eval_mode:
                    state = batch_states[row]
                    cand_states_for_this = batch_cand_states[row]
                    mask_for_this = batch_masks[row]
                    log_prob = log_probs[row]
                    value = values[row].item() if hasattr(values[row], 'item') else float(values[row])

                    # DEBUG: Track before incrementing
                    #old_steps = agent_obj.steps

                    # Push transition to buffer
                    per_env_rollouts[idx].append((
                        state, cand_states_for_this, mask_for_this,
                        a_idx, log_prob, value, reward,
                        1.0 if (terminal_reward != 0.0) else 0.0
                    ))
                    agent_obj.steps += 1

                    # DEBUG: Print every 100 steps
                    #if agent_obj.steps % 100 == 0 or agent_obj.steps <= 10:
                        #print(f"    [TRAIN] Game env {idx}: steps {old_steps}→{agent_obj.steps}, "
                              #f"buffer {agent_obj.buffer.size}/{agent_obj.config.rollout_length}, "
                              #f"updates {agent_obj.updates}, reward {reward:.3f}")

                    # Trigger PPO update if buffer is full
                    #if agent_obj.buffer.is_ready():
                        #print(f"    [UPDATE] Buffer full! Triggering PPO update #{agent_obj.updates + 1}")
                        #agent_obj._ppo_update()
                        #print(f"    [UPDATE] Done! Steps={agent_obj.steps}, Updates={agent_obj.updates}")

                # Check if game over
                done = backgammon.game_over(boards[idx])
                if done:
                    env_active[idx] = False
                    # Flush this env's trajectory to the global PPO buffer (keep it contiguous)
                    for (S_, C_, M_, A_, LP_, V_, R_, D_) in per_env_rollouts[idx]:
                        agent_obj.buffer.push(S_, C_, M_, A_, LP_, V_, R_, D_)
                    per_env_rollouts[idx].clear()
                    # Optionally kick an update here if the global buffer is full
                    if training and not agent_obj.eval_mode and agent_obj.buffer.is_ready():
                        agent_obj._ppo_update()
                    finished += 1
                    # DEBUG: Print who won
                    #agent_won = boards[idx][27] == 15
                    #opp_won = boards[idx][28] == -15
                    #print(f"    [GAME OVER] Env {idx}: Agent pieces borne off={boards[idx][27]}, Opp pieces={boards[idx][28]}, Agent won={agent_won}, Opp won={opp_won}")
                else:
                    # FIX #4: Decrement passes
                    passes_left[idx] -= 1
                    if passes_left[idx] <= 0:
                        # Switch to opponent and roll new dice
                        players[idx] = -1
                        dices[idx] = backgammon.roll_dice()
                        passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

        # ---- Process opponent moves individually ----
        for (idx, dice, board, pmoves, pboards) in opponent_envs:
            # Opponent makes a move
            if opponent == randomAgent:
                # Random opponent
                move = randomAgent.action(board, dice, -1, 0)
            elif opponent == pubeval:
                # Pubeval opponent
                move = pubeval.action(board, dice, -1, 0)
            elif hasattr(opponent, 'action'):
                # Another agent (from pool or self-play)
                move = opponent.action(board, dice, -1, 0, train=False)
            else:
                # Fallback to random
                import random as py_random
                move = pmoves[py_random.randint(0, len(pmoves) - 1)]

            if not _is_empty_move(move):
                boards[idx] = _apply_move_sequence(board, move, -1)

            # Check if game over
            done = backgammon.game_over(boards[idx])
            if done:
                # First: retro-credit opponent win to last agent step (if it happened)
                if training and not agent_obj.eval_mode and boards[idx][28] == -15:
                    loss_reward = -1.0 * agent_obj.config.reward_scale
                    if per_env_rollouts[idx]:
                        (S_, C_, M_, A_, LP_, V_, R_, D_) = per_env_rollouts[idx][-1]
                        per_env_rollouts[idx][-1] = (S_, C_, M_, A_, LP_, V_, R_ + loss_reward, 1.0)
                    else:
                        # Rare: agent never acted – fabricate a 1-step terminal sample in FEATURE space
                        board_pov29 = flip_to_pov_plus1(boards[idx], 1).astype(np.float32)
                        S_feat = one_hot_encoding(board_pov29, False)
                        C_feats = np.zeros((agent_obj.config.max_actions, agent_obj.config.state_dim), dtype=np.float32)
                        C_feats[0] = S_feat
                        M = np.zeros(agent_obj.config.max_actions, dtype=np.float32); M[0] = 1.0
                        per_env_rollouts[idx].append((S_feat, C_feats, M, 0, 0.0, 0.0, loss_reward, 1.0))

                # Now flush this env’s trajectory as one contiguous block
                env_active[idx] = False
                for (S_, C_, M_, A_, LP_, V_, R_, D_) in per_env_rollouts[idx]:
                    agent_obj.buffer.push(S_, C_, M_, A_, LP_, V_, R_, D_)
                per_env_rollouts[idx].clear()
                if training and not agent_obj.eval_mode and agent_obj.buffer.is_ready():
                    agent_obj._ppo_update()
            else:
                # FIX #4: Decrement passes
                passes_left[idx] -= 1
                if passes_left[idx] <= 0:
                    # Switch back to agent and roll new dice
                    players[idx] = 1
                    dices[idx] = backgammon.roll_dice()
                    passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

    return finished


# --- Top-k one-ply lookahead (FIXED: POV-aware) ---
def select_move_with_lookahead(agent_obj, board_pov, dice, i, k=3):
    """
    Select a move using top-k one-ply lookahead.
    IMPORTANT: board_pov must already be flipped so the acting agent is player=+1.
    """
    possible_moves, possible_boards = backgammon.legal_moves(board_pov, dice, player=1)
    nA = len(possible_moves)

    if nA == 0:
        return []
    if nA == 1:
        return possible_moves[0]

    moves_left = 1 + int(dice[0] == dice[1]) - i
    S = agent_obj._encode_state(board_pov, moves_left)

    cand_states = np.stack([
        agent_obj._encode_state(board_after, moves_left - 1)
        for board_after in possible_boards
    ], axis=0)

    if nA > agent_obj.config.max_actions:
        cand_states = cand_states[:agent_obj.config.max_actions]
        possible_moves = possible_moves[:agent_obj.config.max_actions]
        possible_boards = possible_boards[:agent_obj.config.max_actions]
        nA = agent_obj.config.max_actions

    deltas = cand_states - S
    mask = np.ones(nA, dtype=np.float32)

    S_t = torch.as_tensor(S[None, :], dtype=torch.float32, device=agent_obj.device)
    deltas_t = torch.as_tensor(deltas[None, :, :], dtype=torch.float32, device=agent_obj.device)
    mask_t = torch.as_tensor(mask[None, :], dtype=torch.float32, device=agent_obj.device)

    with torch.no_grad():
        logits = agent_obj.acnet.score_moves_delta(S_t, deltas_t, mask_t).squeeze(0)
        top_k = min(k, nA)
        top_k_logits, top_k_indices = torch.topk(logits, top_k)

        best_value = float('-inf')
        best_idx = top_k_indices[0].item()

        for idx in top_k_indices:
            idx = idx.item()
            resulting_board = possible_boards[idx]
            result_state = agent_obj._encode_state(resulting_board, moves_left - 1)
            result_state_t = torch.as_tensor(result_state[None, :], dtype=torch.float32, device=agent_obj.device)
            with torch.no_grad():
                value = agent_obj.acnet.value(result_state_t).item()

            if value > best_value:
                best_value = value
                best_idx = idx

    return possible_moves[best_idx]

def play_one_game(agent1, agent2, training=False, commentary=False,
                  use_lookahead=False, lookahead_k=3):
    """Play one game with optional top-k lookahead (now symmetric)."""
    from ppo_agent import _flip_board, _flip_move

    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1

    if hasattr(agent1, "episode_start"): agent1.episode_start()
    if hasattr(agent2, "episode_start"): agent2.episode_start()

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()
        if commentary:
            print(f"player {player}, dice {dice}")

        # FIX #4: Doubles handling matches training now (in training)
        for r in range(1 + int(dice[0] == dice[1])):
            board_copy = board.copy()

            if player == 1:
                if use_lookahead and not training and hasattr(agent1, '_encode_state'):
                    move = select_move_with_lookahead(agent1, board_copy, dice, i=r, k=lookahead_k)
                else:
                    move = agent1.action(board_copy, dice, player, i=r, train=training)
            else:
                if use_lookahead and not training and hasattr(agent2, '_encode_state'):
                    board_pov = _flip_board(board_copy)
                    move_pov = select_move_with_lookahead(agent2, board_pov, dice, i=r, k=lookahead_k)
                    move = _flip_move(move_pov)
                else:
                    move = agent2.action(board_copy, dice, player, i=r, train=training)

            if _is_empty_move(move):
                continue

            board = _apply_move_sequence(board, move, player)

        player = -player

    winner = -player
    final_board = board

    if hasattr(agent1, "end_episode"):
        agent1.end_episode(+1 if winner == 1 else -1, final_board, perspective=+1)
    if hasattr(agent2, "end_episode"):
        agent2.end_episode(+1 if winner == -1 else -1, final_board, perspective=-1)

    return winner, final_board

def evaluate(agent_mod, evaluation_agent, n_eval, label="", debug_sides=False,
             use_lookahead=False, lookahead_k=3):
    """Evaluate agent with fixed side alternation."""
    wins = 0
    wins_as_p1 = 0
    wins_as_p2 = 0
    games_as_p1 = 0
    games_as_p2 = 0

    # FIX #9: Removed unused rolling_window (was computed incorrectly)

    for g in range(n_eval):
        # Agent 1 plays as player 1
        winner, _ = play_one_game(agent_mod, evaluation_agent, training=False,
                                  use_lookahead=use_lookahead, lookahead_k=lookahead_k)
        if winner == 1:
            wins += 1
            wins_as_p1 += 1
        games_as_p1 += 1

        # Agent 1 plays as player -1
        winner, _ = play_one_game(evaluation_agent, agent_mod, training=False,
                                  use_lookahead=use_lookahead, lookahead_k=lookahead_k)
        if winner == -1:
            wins += 1
            wins_as_p2 += 1
        games_as_p2 += 1

    wr = 100.0 * wins / (n_eval * 2)
    p1_wr = 100.0 * wins_as_p1 / games_as_p1 if games_as_p1 > 0 else 0
    p2_wr = 100.0 * wins_as_p2 / games_as_p2 if games_as_p2 > 0 else 0

    lookahead_str = f" (k={lookahead_k} lookahead)" if use_lookahead else ""
    print(f"{label}{lookahead_str}: {wr:.1f}%", end="")

    if debug_sides:
        print(f"  [P1: {p1_wr:.1f}% | P2: {p2_wr:.1f}%]", end="")

    print()
    return wr


class CheckpointLeague:
    """Manage league of historical checkpoints for evaluation."""
    def __init__(self, checkpoint_dir: Path, agent_module_name: str = "ppo_agent"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.agent_module_name = agent_module_name
        self.checkpoints = {}

        print(f"[League] Initialized at {self.checkpoint_dir}")

    def add_checkpoint(self, game_count: int, checkpoint_path: str):
        """Add checkpoint to league."""
        league_path = self.checkpoint_dir / f"league_{game_count}.pt"
        shutil.copy(checkpoint_path, league_path)
        self.checkpoints[game_count] = league_path
        print(f"[League] Added checkpoint at {game_count:,} games")

    def evaluate_against_league(self, current_agent, n_eval_per_opponent: int = 50):
        """Evaluate current agent against all league members."""
        if not self.checkpoints:
            print("[League] No checkpoints to evaluate against")
            return {}

        results = {}
        agent_mod = __import__(self.agent_module_name)

        print(f"\n[League] Evaluating against {len(self.checkpoints)} checkpoints")
        print("-" * 60)

        for game_count in sorted(self.checkpoints.keys()):
            try:
                checkpoint_path = self.checkpoints[game_count]

                opponent = agent_mod.PPOAgent()
                opponent.load(str(checkpoint_path))
                opponent.set_eval_mode(True)

                wr = evaluate(current_agent, opponent, n_eval_per_opponent,
                            label=f"  vs {game_count:,}g checkpoint", debug_sides=False)
                results[game_count] = wr

            except Exception as e:
                print(f"  Error evaluating vs {game_count:,}g: {e}")
                results[game_count] = None

        print("-" * 60)
        return results

def train(n_games=200_000,
          n_epochs=5_000,
          n_eval=200,
          eval_vs="pubeval",
          model_size='large',
          use_opponent_pool=True,
          pool_snapshot_every=5_000,
          pool_max_size=12,
          pool_sample_rate=0.45,
          pubeval_sample_rate=0.30,
          random_sample_rate=0.25,
          use_eval_lookahead=True,
          eval_lookahead_k=3,
          use_bc_warmstart=False,
          league_checkpoint_every=20_000,
          n_eval_league=50):
    """
    Main training loop with opponent pool and league evaluation.

    Args:
        n_games: Total number of training games
        n_epochs: Evaluate every N games
        n_eval: Number of evaluation games
        eval_vs: Baseline opponent ('pubeval' or 'random')
        model_size: Model size ('small', 'medium', 'large')
        use_opponent_pool: Whether to use opponent pool
        pool_snapshot_every: Add snapshot to pool every N games
        pool_max_size: Maximum pool size
        pool_sample_rate: Probability of sampling from pool
        pubeval_sample_rate: Probability of sampling pubeval
        random_sample_rate: Probability of sampling random
        use_eval_lookahead: Use lookahead during evaluation
        eval_lookahead_k: Lookahead depth for evaluation
        league_checkpoint_every: Add to league every N games
        n_eval_league: Games per league opponent
    """
    print("\n" + "=" * 70)
    print("PPO TRAINING - FIXED VERSION")
    print("=" * 70)
    print(f"Model size: {model_size.upper()}")
    print(f"Total games: {n_games:,}")
    print(f"Evaluate every: {n_epochs:,} games")
    print(f"Evaluation games: {n_eval}")
    print(f"Baseline: {eval_vs}")
    print(f"Lookahead: {use_eval_lookahead} (k={eval_lookahead_k})")
    print(f"Device: {agent.get_device()}")
    print("=" * 70 + "\n")

    # Initialize agent with specified model size
    cfg = agent.get_config(model_size)

    # Gentle hyperparameter tweaks for better curriculum learning
    cfg.ppo_epochs = 3  # Reduced from 4 to reduce overfitting per rollout
    cfg.entropy_min = 0.01  # Keep exploration alive longer

    # FIX #7: Device selection consistency - use CPU for stability
    device = agent.get_device()
    #print(f"\n⚠️  USING CPU DEVICE (for PPO stability)")
    #print(f"   Expected: 2-3x slower than MPS/GPU, but more stable training\n")

    agent_instance = agent.PPOAgent(config=cfg, device=device)

    # Set checkpoint path based on model size
    checkpoint_base_path = Path(CHECKPOINT_DIR)
    _ensure_dir(checkpoint_base_path)
    agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_ppo_{model_size}.pt"

    # Initialize opponent pool
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{model_size}"
        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name="ppo_agent",
            max_size=pool_max_size,
            seed=42,
            device=device  # Use same device as training agent
        )
        print(f"\nOpponent Pool Configuration:")
        print(f"  Snapshot every: {pool_snapshot_every:,} games")
        print(f"  Max size: {pool_max_size}")
        print(f"  Sample rates: Pool={pool_sample_rate:.0%}, Pubeval={pubeval_sample_rate:.0%}, Random={random_sample_rate:.0%}")

        # Verify existing pool if any
        if len(opponent_pool) > 0:
            print(f"\n{'='*60}")
            print("EXISTING OPPONENT POOL FOUND")
            print(f"{'='*60}")
            opponent_pool.verify_snapshots()
            print(f"{'='*60}\n")
        else:
            print(f"  Pool is empty - will seed with initial agent")

            # --- Seed the pool immediately so pool% > 0 from the start ---
            print(f"\n{'='*60}")
            print("SEEDING OPPONENT POOL")
            print(f"{'='*60}")
            latest_path = agent.CHECKPOINT_PATH.parent / f"latest_ppo_{model_size}.pt"
            agent_instance.save(str(latest_path))
            opponent_pool.add_snapshot(latest_path, label="(seed at 0 games)")
            print(f"✓ Seed snapshot added")
            print(f"{opponent_pool.get_pool_info()}")
            print(f"{'='*60}\n")
        print()

    # Initialize checkpoint league
    league = CheckpointLeague(
        checkpoint_dir=checkpoint_base_path / f"league_{model_size}",
        agent_module_name="ppo_agent"
    )

    # Baseline opponent
    if eval_vs == "pubeval":
        baseline = pubeval
    else:
        baseline = randomAgent

    # Performance tracking
    perf_data = {
        'vs_baseline': [],
        'vs_baseline_lookahead': [],
        'vs_random': [],
        'vs_league_avg': [],
        'vs_latest_checkpoint': []
    }

    # FIX #6: Removed best_wr variable (was never updated)

    # Opponent statistics
    opponent_stats = {
        'self_play': 0,
        'pool': 0,
        'pool_best': 0,
        'pubeval': 0,
        'random': 0
    }

    # Training loop
    train_bar = tqdm(total=n_games, desc="Training", unit="game")
    games_done = 0

    # DEBUG: Initial state
    #print(f"\n[DEBUG] Initial agent state:")
    #print(f"  steps: {agent_instance.steps}")
    #print(f"  updates: {agent_instance.updates}")
    #print(f"  buffer size: {agent_instance.buffer.size}")
    #print(f"  rollout_length: {agent_instance.config.rollout_length}")
    #print(f"  eval_mode: {agent_instance.eval_mode}")
    #print(f"  Training: {not agent_instance.eval_mode}")
    #print()

    # Schedule thresholds (not modulo-based anymore)
    next_eval_at = n_epochs
    next_snapshot_at = pool_snapshot_every

    # Pool curriculum ramping parameters
    pool_start_games = 5_000        # No pool before this - build competence first
    pool_ramp_end = 50_000          # Linearly ramp pool% up to target by here
    pool_target_rate = 0.30         # Desired steady-state pool usage

    # Track best checkpoint for stable curriculum
    best_wr_vs_random = 0.0
    best_ckpt_path = checkpoint_base_path / f"best_so_far_{model_size}.pt"

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Opponent Pool: {'ENABLED' if use_opponent_pool else 'DISABLED'}")
    if use_opponent_pool and opponent_pool:
        print(f"  Initial pool size: {len(opponent_pool)}")
        print(f"  Pool ramp: 0% @ {pool_start_games:,} → {pool_target_rate*100:.0f}% @ {pool_ramp_end:,} games")
        print(f"  First snapshot at: {next_snapshot_at:,} games")
        print(f"  Snapshot frequency: every {pool_snapshot_every:,} games")
    print(f"Evaluation: every {n_epochs:,} games")
    print(f"Total games: {n_games:,}")
    print(f"{'='*60}\n")

    # Optional: Behavior cloning warm-start
    if use_bc_warmstart:
        agent.warmstart_with_pubeval(batch_iter, epochs=3, assume_second_roll=False)

    while games_done < n_games:
        # Calculate effective pool rate based on curriculum ramp
        if use_opponent_pool and opponent_pool and len(opponent_pool) > 0:
            if games_done < pool_start_games:
                effective_pool_rate = 0.0  # No pool pressure early
            elif games_done < pool_ramp_end:
                # Linear ramp from 0% to target%
                t = (games_done - pool_start_games) / (pool_ramp_end - pool_start_games)
                effective_pool_rate = pool_target_rate * max(0.0, min(1.0, t))
            else:
                effective_pool_rate = pool_target_rate  # Full pool pressure
        else:
            effective_pool_rate = 0.0

        # Choose opponent based on ramped curriculum
        opponent = None
        opponent_type = 'self_play'

        r = random.random()
        if r < effective_pool_rate:
            # Sample from pool - bias toward older snapshots early, recent later
            bias_recent = games_done >= pool_ramp_end

            # 50% chance to use best-so-far for stability
            if random.random() < 0.5 and best_ckpt_path.exists():
                try:
                    agent_mod = importlib.import_module("ppo_agent")
                    opponent = agent_mod.PPOAgent(device=device)
                    opponent.load(str(best_ckpt_path), map_location=device)
                    opponent.set_eval_mode(True)
                    opponent_type = 'pool_best'
                except Exception as e:
                    # Fall back to regular pool sampling
                    opponent = opponent_pool.sample_opponent(bias_recent=bias_recent)
                    opponent_type = 'pool' if opponent is not None else 'self_play'
            else:
                opponent = opponent_pool.sample_opponent(bias_recent=bias_recent)
                opponent_type = 'pool' if opponent is not None else 'self_play'

            # Debug: Log pool usage occasionally
            #if games_done % 1000 == 0 and opponent_type != 'self_play':
            #    ramp_pct = effective_pool_rate * 100
            #    print(f"  [Pool] Sampling at {ramp_pct:.1f}% rate (target: {pool_target_rate*100:.0f}%)")
        elif r < effective_pool_rate + pubeval_sample_rate:
            opponent = pubeval
            opponent_type = 'pubeval'
        elif r < effective_pool_rate + pubeval_sample_rate + random_sample_rate:
            opponent = randomAgent
            opponent_type = 'random'
        # else: self-play (remaining probability)
        if opponent is None:
            opponent = agent_instance
            opponent_type = 'self_play'

        # Run one batched rollout of complete games
        finished = play_games_batched(agent_instance, opponent, batch_size=BATCH_SIZE, training=True)

        # Clamp so we never exceed target (in case last batch overshoots)
        if games_done + finished > n_games:
            finished = n_games - games_done

        # Accounting
        games_done += finished
        train_bar.update(finished)
        opponent_stats[opponent_type] += finished

        # DEBUG: After first batch
        #if games_done == BATCH_SIZE:
            #print(f"\n[DEBUG] After first batch ({BATCH_SIZE} games):")
            #print(f"  steps: {agent_instance.steps} (expected ~{BATCH_SIZE * 30})")
            #print(f"  updates: {agent_instance.updates}")
            #print(f"  buffer size: {agent_instance.buffer.size}")
            #if agent_instance.steps == 0:
            #    print(f"  ⚠️  WARNING: No steps after {BATCH_SIZE} games!")
            #    print(f"  ⚠️  This means agent moves are not being recorded!")
            #print()

        # Update bar postfix periodically
        if games_done % 100 == 0 or games_done == n_games:
            postfix = {"steps": f"{agent_instance.steps:,}", "upd": agent_instance.updates}

            # Add loss values from recent updates
            if hasattr(agent_instance, 'rollout_stats'):
                stats = agent_instance.rollout_stats
                if stats['policy_loss'] and len(stats['policy_loss']) > 0:
                    postfix["πL"] = f"{stats['policy_loss'][-1]:.3f}"  # Policy loss
                if stats['value_loss'] and len(stats['value_loss']) > 0:
                    postfix["VL"] = f"{stats['value_loss'][-1]:.3f}"  # Value loss
                if stats['grad_norm'] and len(stats['grad_norm']) > 0:
                    postfix["∇"] = f"{stats['grad_norm'][-1]:.2f}"  # Grad norm

            total_opp_games = sum(opponent_stats.values())
            if total_opp_games > 0:
                postfix["self%"] = f"{100.0 * opponent_stats['self_play'] / total_opp_games:.0f}"
                postfix["pool%"] = f"{100.0 * opponent_stats['pool'] / total_opp_games:.0f}"
                postfix["pub%"]  = f"{100.0 * opponent_stats['pubeval'] / total_opp_games:.0f}"
                postfix["rnd%"]  = f"{100.0 * opponent_stats['random'] / total_opp_games:.0f}"
            train_bar.set_postfix(postfix)

        # Snapshot by threshold (used to be g % pool_snapshot_every)
        while use_opponent_pool and opponent_pool and games_done >= next_snapshot_at and next_snapshot_at > 0:
            print(f"\n{'='*60}")
            print(f"SNAPSHOT CHECKPOINT at {games_done:,} games")
            print(f"{'='*60}")

            latest_path = agent.CHECKPOINT_PATH.parent / f"latest_ppo_{model_size}.pt"
            agent_instance.save(str(latest_path))

            # Verify the checkpoint was actually saved
            if not latest_path.exists():
                print(f"❌ ERROR: Checkpoint not saved at {latest_path}")
                print(f"   Check that {latest_path.parent} is writable")
            else:
                print(f"✓ Checkpoint saved: {latest_path}")
                print(f"  File size: {latest_path.stat().st_size / 1024:.1f} KB")

            # Gate snapshot addition: only add if reasonably competent
            print(f"\n🔍 Checking competence before adding to pool...")
            agent_instance.set_eval_mode(True)
            wr_vs_random = evaluate(agent_instance, randomAgent, n_eval=40,
                                   label="vs random (gate check)", debug_sides=False,
                                   use_lookahead=False)
            agent_instance.set_eval_mode(False)

            min_competence = 60.0  # Minimum win rate vs random to add to pool
            if wr_vs_random >= min_competence:
                print(f"✓ Snapshot meets competence threshold ({wr_vs_random:.1f}% ≥ {min_competence}%)")
                opponent_pool.add_snapshot(latest_path, label=f"(after {next_snapshot_at:,} games, WR={wr_vs_random:.0f}%)")
                print(f"\n{opponent_pool.get_pool_info()}")
            else:
                print(f"❌ Snapshot rejected ({wr_vs_random:.1f}% < {min_competence}%)")
                print(f"   Not adding to pool to avoid polluting curriculum")

            # Verify pool can actually load snapshots
            if len(opponent_pool) > 0 and games_done == next_snapshot_at:  # First snapshot
                print("\n🔍 Testing pool loading...")
                test_opp = opponent_pool.sample_opponent(bias_recent=True)
                if test_opp is not None:
                    print(f"✓ Successfully loaded opponent from pool!")
                    print(f"  Opponent steps: {test_opp.steps if hasattr(test_opp, 'steps') else 'N/A'}")
                else:
                    print(f"❌ WARNING: Could not load opponent from pool")
                    print(f"   Pool will not be used in training!")

            print(f"{'='*60}\n")
            next_snapshot_at += pool_snapshot_every  # schedule next threshold

        # Evaluate by threshold (used to be if (g % n_epochs) == 0)
        while games_done >= next_eval_at and next_eval_at > 0:
            print()
            agent_instance.set_eval_mode(True)
            print("[Eval] Agent in EVAL mode")
            print(f"\n--- Evaluation after {next_eval_at:,} games ---")

            wr = evaluate(agent_instance, baseline, n_eval,
                        label=f"vs {eval_vs} (greedy)",
                        debug_sides=False, use_lookahead=False)
            perf_data['vs_baseline'].append(wr)

            if use_eval_lookahead:
                wr_lookahead = evaluate(agent_instance, baseline, min(100, n_eval),
                                        label=f"vs {eval_vs}",
                                        debug_sides=False, use_lookahead=True,
                                        lookahead_k=eval_lookahead_k)
                perf_data['vs_baseline_lookahead'].append(wr_lookahead)
                print(f"  Lookahead improvement: +{(wr_lookahead - wr):.1f}% points")

            try:
                wr_rand = evaluate(agent_instance, randomAgent, max(50, n_eval // 2),
                                label="vs random", debug_sides=False, use_lookahead=False)
                perf_data['vs_random'].append(wr_rand)

                # Track and save best checkpoint
                if wr_rand > best_wr_vs_random:
                    best_wr_vs_random = wr_rand
                    agent_instance.save(str(best_ckpt_path))
                    print(f"  🌟 NEW BEST vs random: {wr_rand:.1f}% (saved to {best_ckpt_path.name})")

                if wr_rand < 50.0:
                    print(f"  ⚠️  WARNING: Only {wr_rand:.1f}% vs random!")
            except Exception as e:
                print(f"[Eval] Skipped vs-random: {e}")

            agent_instance.set_eval_mode(False)
            next_eval_at += n_epochs  # schedule next threshold

    train_bar.close()


    # Final statistics
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Model size: {model_size.upper()}")
    print()
    print("Opponent Usage:")
    total_games = sum(opponent_stats.values())
    if total_games > 0:
        for opp_type, count in opponent_stats.items():
            pct = 100.0 * count / total_games
            print(f"  {opp_type.capitalize()}: {count:,} ({pct:.1f}%)")

    print()
    # FIX #6: Removed misleading best_wr printout
    print(f"Final best win-rate vs random: {best_wr_vs_random:.1f}%")
    if perf_data['vs_league_avg']:
        print(f"Final league average: {perf_data['vs_league_avg'][-1]:.1f}%")

    print()
    print("Final PPO State:")
    print(f"  Updates: {agent_instance.updates}")
    print(f"  Steps: {agent_instance.steps}")
    print(f"  Entropy coef: {agent_instance.current_entropy_coef:.4f}")

    if hasattr(agent_instance, 'rollout_stats'):
        stats = agent_instance.rollout_stats
        if stats['nA_values']:
            print(f"  Avg legal actions: {np.mean(stats['nA_values'][-1000:]):.1f}")
        if stats['masked_entropy']:
            print(f"  Final entropy: {np.mean(stats['masked_entropy'][-10:]):.4f}")

    print("=" * 70)

    plot_perf(perf_data, title=f"PPO Training ({model_size.upper()} model)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent for backgammon')
    parser.add_argument('--model-size', type=str, default='large',
                       choices=['small', 'medium', 'large'],
                       help='Model size: small (CPU), medium (M1/M2/T4 GPU), large (good GPU)')
    parser.add_argument('--n-games', type=int, default=200_000,
                       help='Total number of training games')
    parser.add_argument('--n-epochs', type=int, default=5_000,
                       help='Evaluate every N games')
    parser.add_argument('--cpu-test', action='store_true',
                       help='Quick CPU test: small model, 10k games, frequent evals')

    args = parser.parse_args()

    # CPU test mode: fast settings for testing
    if args.cpu_test:
        print("\n" + "=" * 70)
        print("CPU TEST MODE")
        print("=" * 70)
        print("  Model: small")
        print("  Games: 50,000")
        print("  Eval every: 5,000 games")
        print("  Eval games: 200")
        print("  Baseline: pubeval")
        print("  Lookahead: enabled")
        print("=" * 70 + "\n")

        train(
            n_games=50_000,
            n_epochs=5_000,
            n_eval=200,
            eval_vs="pubeval",
            model_size='small',
            use_opponent_pool=False,  # DISABLED - build competence first!
            pool_snapshot_every=5_000,
            pubeval_sample_rate=0.20,  # 20% pubeval exposure
            random_sample_rate=0.20,   # 20% random for robustness
            use_eval_lookahead=True,
            eval_lookahead_k=3,
            use_bc_warmstart=True,  # Jump-start with pubeval imitation
            league_checkpoint_every=10_000,
        )
    else:
        train(
            n_games=args.n_games,
            n_epochs=args.n_epochs,
            n_eval=200,
            eval_vs="pubeval",
            model_size=args.model_size,
            # Improved curriculum - reduced random opponent
            use_opponent_pool=True,
            pool_snapshot_every=5_000,
            pool_max_size=12,
            pool_sample_rate=0.50,      # 50% pool (self-play curriculum)
            pubeval_sample_rate=0.45,   # 45% pubeval (strong baseline)
            random_sample_rate=0.05,    # 5% random (robustness only)
            # Evaluation
            use_eval_lookahead=True,
            eval_lookahead_k=3,
        )
