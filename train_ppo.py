#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-Optimized PPO Training Script

NEW: Support for small/medium/large model sizes
- Small: ~80K params, 5-10x faster, great for CPU testing
- Medium: ~400K params, 2-3x faster, good for T4 GPU
- Large: ~1.6M params, best performance, needs good GPU

All fixes from previous versions included.
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

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
import ppo_agent as agent
from opponent_pool import OpponentPool

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Google Drive paths
GDRIVE_ROOT = "/content/drive/MyDrive/Backgammon" if os.path.exists("/content/drive") else "."
CHECKPOINT_DIR = os.path.join(GDRIVE_ROOT, "checkpoints")

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
    plt.show()

def _is_empty_move(move):
    if move is None: return True
    if isinstance(move, (list, tuple)): return len(move) == 0
    if isinstance(move, np.ndarray): return move.size == 0
    return False

def _apply_move_sequence(board, move_seq, player):
    mv = np.asarray(move_seq, dtype=np.int32)
    return backgammon.update_board(board, mv, player)

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
            value = agent_obj.acnet.value_head(result_state_t).item()
            
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
    
    rolling_window = []
    window_size = 20
    
    lookahead_str = f" (k={lookahead_k} lookahead)" if use_lookahead else ""
    eval_bar = tqdm(range(n_eval), desc=f"Evaluating {label}{lookahead_str}", leave=False, ncols=120)
    
    for i in eval_bar:
        if i % 2 == 0:
            w, _ = play_one_game(agent_mod, evaluation_agent, training=False, 
                                commentary=False, use_lookahead=use_lookahead, 
                                lookahead_k=lookahead_k)
            games_as_p1 += 1
            if w == 1:
                wins += 1
                wins_as_p1 += 1
        else:
            w, _ = play_one_game(evaluation_agent, agent_mod, training=False, 
                                commentary=False, use_lookahead=use_lookahead,
                                lookahead_k=lookahead_k)
            w = -w
            games_as_p2 += 1
            if w == 1:
                wins += 1
                wins_as_p2 += 1
        
        rolling_window.append(1 if w == 1 else 0)
        if len(rolling_window) > window_size:
            rolling_window.pop(0)
        
        current_wr = round(wins / (i + 1) * 100.0, 1)
        rolling_wr = round(100.0 * sum(rolling_window) / len(rolling_window), 1) if rolling_window else 0.0
        
        eval_bar.set_postfix({"WR": f"{current_wr}%", f"Roll{window_size}": f"{rolling_wr}%"})
    
    winrate = round(wins / n_eval * 100.0, 3)
    
    print(f"[Eval] {label}{lookahead_str}: win-rate = {winrate}% over {n_eval} games")
    
    if debug_sides:
        wr_as_p1 = round(100.0 * wins_as_p1 / games_as_p1, 1) if games_as_p1 > 0 else 0.0
        wr_as_p2 = round(100.0 * wins_as_p2 / games_as_p2, 1) if games_as_p2 > 0 else 0.0
        print(f"  Side split: as P1: {wr_as_p1}% ({wins_as_p1}/{games_as_p1}), "
              f"as P2: {wr_as_p2}% ({wins_as_p2}/{games_as_p2})")
        
        if abs(wr_as_p1 - wr_as_p2) > 20.0:
            print(f"  ⚠️  WARNING: Large side discrepancy!")
    
    return winrate


class CheckpointLeague:
    """Manages historical checkpoints for self-play evaluation."""
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR, league_size=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.league_dir = self.checkpoint_dir / "league"
        self.league_dir.mkdir(exist_ok=True)
        
        self.league_size = league_size
        self.checkpoints = []
        
        self._load_existing_checkpoints()
        
        print(f"Checkpoint League initialized")
        print(f"  Directory: {self.league_dir}")
        print(f"  League size: {league_size}")
        print(f"  Existing checkpoints: {len(self.checkpoints)}")
    
    def _load_existing_checkpoints(self):
        if not self.league_dir.exists():
            return
        
        for checkpoint_file in sorted(self.league_dir.glob("checkpoint_g*.pt")):
            try:
                game_num = int(checkpoint_file.stem.split('_g')[1])
                self.checkpoints.append((game_num, checkpoint_file))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse checkpoint {checkpoint_file.name}")
        
        if len(self.checkpoints) > self.league_size:
            self.checkpoints = sorted(self.checkpoints, key=lambda x: x[0])[-self.league_size:]
    
    def add_checkpoint(self, game_num, source_path):
        checkpoint_name = f"checkpoint_g{game_num}.pt"
        checkpoint_path = self.league_dir / checkpoint_name
        
        shutil.copy(source_path, checkpoint_path)
        self.checkpoints.append((game_num, checkpoint_path))
        
        if len(self.checkpoints) > self.league_size:
            old_game_num, old_path = self.checkpoints.pop(0)
            try:
                old_path.unlink()
                print(f"Removed old checkpoint from league: g{old_game_num}")
            except Exception as e:
                print(f"Could not remove old checkpoint: {e}")
        
        print(f"Added checkpoint g{game_num} to league (size: {len(self.checkpoints)}/{self.league_size})")
    
    def evaluate_against_league(self, current_agent, n_eval_per_opponent=100):
        if not self.checkpoints:
            print("League is empty, skipping evaluation")
            return {}
        
        print(f"\nEvaluating against {len(self.checkpoints)} league opponents:")
        results = {}
        
        for game_num, checkpoint_path in self.checkpoints:
            print(f"\n  vs checkpoint g{game_num}...")
            
            try:
                opponent = agent.PPOAgent()
                opponent.load(str(checkpoint_path), map_location=opponent.device)
                opponent.set_eval_mode(True)
                
                wr = evaluate(
                    current_agent, 
                    opponent, 
                    n_eval_per_opponent,
                    label=f"vs league g{game_num}",
                    debug_sides=True
                )
                results[game_num] = wr
                
            except Exception as e:
                print(f"  Error evaluating vs g{game_num}: {e}")
                results[game_num] = None
        
        return results


def train(
    n_games=200_000,
    n_epochs=5_000,
    n_eval=200,
    eval_vs="pubeval",
    league_checkpoint_every=25_000,
    league_size=5,
    n_eval_league=100,
    use_gdrive=True,
    # NEW: Model size selection
    model_size='large',  # 'small', 'medium', or 'large'
    # Opponent pool config
    use_opponent_pool=True,
    pool_snapshot_every=5_000,
    pool_max_size=12,
    pool_sample_rate=0.45,
    pubeval_sample_rate=0.30,
    random_sample_rate=0.25,
    # Evaluation enhancements
    use_eval_lookahead=True,
    eval_lookahead_k=3,
):
    """
    PPO training with configurable model size.
    
    NEW FEATURE: model_size parameter
    - 'small': ~80K params, 5-10x faster, for CPU
    - 'medium': ~400K params, 2-3x faster, for T4 GPU
    - 'large': ~1.6M params, best performance, for good GPUs
    """
    
    baseline = pubeval if eval_vs == "pubeval" else randomAgent
    
    checkpoint_base_path = Path(CHECKPOINT_DIR) if use_gdrive else Path("./checkpoints")
    checkpoint_base_path.mkdir(parents=True, exist_ok=True)
    
    if not _ensure_dir(checkpoint_base_path):
        print(f"[WARNING] Could not verify write access to {checkpoint_base_path}")
    
    # NEW: Create agent with specified model size
    print()
    print("=" * 70)
    print("CREATING AGENT")
    print("=" * 70)
    
    # Get config for specified size
    if hasattr(agent, 'get_config'):
        cfg = agent.get_config(model_size)
        agent_instance = agent.PPOAgent(config=cfg)
        # Update module-level agent
        agent._default_agent = agent_instance
    else:
        print(f"WARNING: get_config() not found in ppo_agent.")
        print(f"Using default config. To enable model size selection,")
        print(f"add the config classes from model_size_configs.py to ppo_agent.py")
        agent_instance = agent._get_agent()
    
    # Set checkpoint path
    agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_ppo_{model_size}.pt"
    
    print("=" * 70)
    print()
    
    # Initialize opponent pool
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{model_size}"
        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name="ppo_agent",
            max_size=pool_max_size,
            seed=42
        )
    
    league = CheckpointLeague(checkpoint_dir=checkpoint_base_path, league_size=league_size)
    
    perf_data = {
        'vs_baseline': [],
        'vs_baseline_lookahead': [],
        'vs_random': [],
        'vs_league_avg': [],
        'vs_latest_checkpoint': [],
    }
    best_wr = -1
    
    opponent_stats = {
        'pool': 0,
        'pubeval': 0,
        'random': 0,
        'self': 0,
    }
    
    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model size: {model_size.upper()}")
    print(f"Total games: {n_games:,}")
    print(f"Evaluation every: {n_epochs:,} games")
    print(f"Baseline: {eval_vs}")
    print()
    print("OPPONENT CURRICULUM:")
    print(f"  Pool: {pool_sample_rate*100:.0f}%")
    print(f"  Pubeval: {pubeval_sample_rate*100:.0f}%")
    print(f"  Random: {random_sample_rate*100:.0f}%")
    print()
    if use_eval_lookahead:
        print(f"EVAL LOOKAHEAD: Enabled (k={eval_lookahead_k})")
    print("=" * 70)
    print()
    
    if hasattr(agent, "set_eval_mode"):
        agent.set_eval_mode(False)
        print("[Training] Agent in TRAINING mode")
    
    train_bar = tqdm(range(n_games), desc="Training", ncols=120)
    
    for g in train_bar:
        # Select opponent
        opponent_type = None
        if use_opponent_pool and opponent_pool and len(opponent_pool) > 0:
            rand = np.random.random()
            if rand < pool_sample_rate:
                opponent = opponent_pool.sample_opponent(bias_recent=True)
                opponent_stats['pool'] += 1
                opponent_type = 'pool'
            elif rand < pool_sample_rate + pubeval_sample_rate:
                opponent = pubeval
                opponent_stats['pubeval'] += 1
                opponent_type = 'pubeval'
            else:
                opponent = randomAgent
                opponent_stats['random'] += 1
                opponent_type = 'random'
        else:
            opponent = agent
            opponent_stats['self'] += 1
            opponent_type = 'self'
        
        # Play training game
        winner, final_board = play_one_game(agent, opponent, training=True, commentary=False)

        # Update progress bar
        if g % 100 == 0:
            postfix = {
                "steps": getattr(agent, '_steps', 0),
                "updates": getattr(agent, '_updates', 0),
                "ent": f"{getattr(agent, '_current_entropy_coef', agent.CFG.entropy_coef):.4f}",
            }
            
            if hasattr(agent, '_rollout_stats'):
                stats = agent._rollout_stats
                if stats['nA_values']:
                    recent_nA = stats['nA_values'][-100:]
                    postfix["nA"] = f"{np.mean(recent_nA):.1f}"
            
            if sum(opponent_stats.values()) > 0:
                total = sum(opponent_stats.values())
                pub_pct = 100.0 * opponent_stats['pubeval'] / total
                postfix["pub%"] = f"{pub_pct:.0f}"
            
            train_bar.set_postfix(postfix)
        
        # Add snapshot to opponent pool
        if use_opponent_pool and opponent_pool and (g % pool_snapshot_every) == 0 and g > 0:
            latest_path = agent.CHECKPOINT_PATH.parent / f"latest_ppo_{model_size}.pt"
            agent.save(str(latest_path))
            opponent_pool.add_snapshot(latest_path, label=f"(after {g:,} games)")
            print(f"\n{opponent_pool.get_pool_info()}")

        # Regular evaluation
        if (g % n_epochs) == 0:
            print()
            
            if hasattr(agent, "set_eval_mode"): 
                agent.set_eval_mode(True)
                print("[Eval] Agent in EVAL mode")
            
            print(f"\n--- Evaluation after {g:,} games ---")
            
            # Greedy evaluation
            wr = evaluate(agent, baseline, n_eval, 
                         label=f"vs {eval_vs} (greedy)", 
                         debug_sides=True,
                         use_lookahead=False)
            perf_data['vs_baseline'].append(wr)
            
            # Lookahead evaluation
            if use_eval_lookahead:
                wr_lookahead = evaluate(agent, baseline, min(100, n_eval), 
                                       label=f"vs {eval_vs}", 
                                       debug_sides=True,
                                       use_lookahead=True,
                                       lookahead_k=eval_lookahead_k)
                perf_data['vs_baseline_lookahead'].append(wr_lookahead)
                
                improvement = wr_lookahead - wr
                print(f"  Lookahead improvement: +{improvement:.1f}% points")
            
            # Random sanity check
            try:
                wr_rand = evaluate(agent, randomAgent, max(50, n_eval // 2),
                                 label="vs random",
                                 debug_sides=True,
                                 use_lookahead=False)
                perf_data['vs_random'].append(wr_rand)
                
                if wr_rand < 50.0:
                    print(f"  ⚠️  WARNING: Only {wr_rand:.1f}% vs random!")
            except Exception as e:
                print(f"[Eval] Skipped vs-random: {e}")

            # PPO stats
            print(f"[PPO] Updates: {agent._updates}, Entropy: {agent._current_entropy_coef:.4f}")
            
            if hasattr(agent, '_rollout_stats'):
                stats = agent._rollout_stats
                stats_list = []
                if stats['nA_values']:
                    stats_list.append(f"Avg nA: {np.mean(stats['nA_values'][-500:]):.1f}")
                if stats['masked_entropy']:
                    stats_list.append(f"Entropy: {np.mean(stats['masked_entropy'][-10:]):.4f}")
                if stats['advantages']:
                    stats_list.append(f"|Adv|: {np.mean(np.abs(stats['advantages'][-500:])):.3f}")
                if stats_list:
                    print(f"  Stats: {', '.join(stats_list)}")

            # Save checkpoints
            latest_path = agent.CHECKPOINT_PATH.parent / f"latest_ppo_{model_size}.pt"
            _safe_save_agent(agent, latest_path, label="latest")
            
            if hasattr(agent, "save") and wr > best_wr:
                _safe_save_agent(agent, agent.CHECKPOINT_PATH, label="best")
                best_wr = wr
                print(f"[Checkpoint] New best: {best_wr:.3f}%")
            else:
                print(f"[Checkpoint] Best: {best_wr:.3f}%")

            if hasattr(agent, "set_eval_mode"): 
                agent.set_eval_mode(False)
                print("[Training] Back to TRAINING mode")
            
            print()
        
        # League evaluation
        if (g % league_checkpoint_every) == 0 and g > 0:
            print()
            print("=" * 70)
            print(f"LEAGUE CHECKPOINT at {g:,} games")
            print("=" * 70)
            
            temp_league_path = Path(checkpoint_base_path) / f"temp_league_{model_size}.pt"
            agent.save(str(temp_league_path))
            league.add_checkpoint(g, str(temp_league_path))
            
            if hasattr(agent, "set_eval_mode"): 
                agent.set_eval_mode(True)
            
            league_results = league.evaluate_against_league(agent, n_eval_per_opponent=n_eval_league)
            
            if league_results:
                valid_results = [v for v in league_results.values() if v is not None]
                if valid_results:
                    avg_league_wr = np.mean(valid_results)
                    perf_data['vs_league_avg'].append(avg_league_wr)
                    
                    most_recent = max(league_results.keys())
                    perf_data['vs_latest_checkpoint'].append(league_results[most_recent])
                    
                    print(f"\nAverage win-rate vs league: {avg_league_wr:.1f}%")
            
            if hasattr(agent, "set_eval_mode"): 
                agent.set_eval_mode(False)
            
            print()

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
    print(f"Final best win-rate vs {eval_vs}: {best_wr:.3f}%")
    if perf_data['vs_league_avg']:
        print(f"Final league average: {perf_data['vs_league_avg'][-1]:.1f}%")
    
    print()
    print("Final PPO State:")
    print(f"  Updates: {agent._updates}")
    print(f"  Steps: {agent._steps}")
    print(f"  Entropy coef: {agent._current_entropy_coef:.4f}")
    
    if hasattr(agent, '_rollout_stats'):
        stats = agent._rollout_stats
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
                       help='Model size: small (CPU), medium (T4 GPU), large (good GPU)')
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
        print("  Games: 10,000")
        print("  Eval every: 1,000 games")
        print("  Eval games: 50 (faster)")
        print("  Baseline: pubeval")
        print("  Lookahead: disabled (faster)")
        print("=" * 70 + "\n")
        
        train(
            n_games=10_000,
            n_epochs=1_000,
            n_eval=50,  # Fewer eval games for speed
            eval_vs="pubeval",  # Still measure vs pubeval!
            model_size='small',
            use_eval_lookahead=False,  # Disable for speed
            pool_snapshot_every=2_000,
            league_checkpoint_every=5_000,
        )
    else:
        train(
            n_games=args.n_games,
            n_epochs=args.n_epochs,
            n_eval=200,
            eval_vs="pubeval",
            model_size=args.model_size,
            # Improved curriculum
            use_opponent_pool=True,
            pool_snapshot_every=5_000,
            pool_max_size=12,
            pool_sample_rate=0.45,
            pubeval_sample_rate=0.30,
            random_sample_rate=0.25,
            # Evaluation
            use_eval_lookahead=True,
            eval_lookahead_k=3,
        )
