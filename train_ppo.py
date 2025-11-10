#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Training Script with MPS Support - FIXED VERSION

Support for small/medium/large model sizes
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
from dataclasses import dataclass, field

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
import ppo_agent as agent
from opponent_pool import OpponentPool
from utils import _ensure_dir, _safe_save_agent, plot_perf, _is_empty_move, _apply_move_sequence, flip_to_pov_plus1, flip_to_pov_plus1, get_device
from play_games import play_games_batched
from evaluate import evaluate, CheckpointLeague

# Set seeds for reproducibility
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Local checkpoint directory (no Google Drive)
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 8  # or 16/32 for more parallel games

print(f"Batch self-play games: {BATCH_SIZE} games in parallel")

# =========================
# Core training structures
# =========================

@dataclass
class TrainingState:
    # Config / static
    model_size: str
    device: str
    eval_vs: str
    n_games: int
    n_epochs: int
    n_eval: int
    use_opponent_pool: bool
    pool_snapshot_every: int
    pool_max_size: int
    pool_sample_rate: float
    pubeval_sample_rate: float
    random_sample_rate: float
    use_eval_lookahead: bool
    eval_lookahead_k: int
    use_bc_warmstart: bool
    league_checkpoint_every: int
    n_eval_league: int

    # Curriculum tuning
    pool_start_games: int = 5_000
    pool_ramp_end: int = 50_000
    pool_target_rate: float = 0.30

    # Mutable runtime state
    agent_instance: "agent.PPOAgent" = None
    checkpoint_base_path: Path = None
    best_ckpt_path: Path = None
    opponent_pool: "OpponentPool" = None
    league: "CheckpointLeague" = None
    baseline: object = None
    games_done: int = 0
    next_eval_at: int = 0
    next_snapshot_at: int = 0
    best_wr: float = 0.0
    perf_data: dict = field(default_factory=lambda: {
        'vs_baseline': [],
        'vs_baseline_lookahead': [],
        'vs_random': [],
        'vs_league_avg': [],
        'vs_latest_checkpoint': []
    })
    opponent_stats: dict = field(default_factory=lambda: {
        'self_play': 0,
        'pool': 0,
        'pool_best': 0,
        'pubeval': 0,
        'random': 0
    })


# =========================
# Initialization
# =========================

def initialize_training(
    *,
    n_games=200_000,
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
    n_eval_league=50,
    device='cpu'
) -> TrainingState:
    print("\n" + "=" * 70)
    print("PPO TRAINING")
    print("=" * 70)
    print(f"Model size: {model_size.upper()}")
    print(f"Total games: {n_games:,}")
    print(f"Evaluate every: {n_epochs:,} games")
    print(f"Evaluation games: {n_eval}")
    print(f"Baseline: {eval_vs}")
    print(f"Lookahead: {use_eval_lookahead} (k={eval_lookahead_k})")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Agent + gentle curriculum tweaks
    cfg = agent.get_config(model_size)
    cfg.ppo_epochs = 3
    cfg.entropy_min = 0.01
    agent_instance = agent.PPOAgent(config=cfg, device=device)

    # Checkpoint paths
    checkpoint_base_path = Path(CHECKPOINT_DIR)
    _ensure_dir(checkpoint_base_path)
    agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_ppo_{model_size}.pt"
    best_ckpt_path = checkpoint_base_path / f"best_so_far_{model_size}.pt"

    # Opponent pool (optional)
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{model_size}"
        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name="ppo_agent",
            max_size=pool_max_size,
            seed=RANDOM_SEED,
            device=device
        )
        print(f"\nOpponent Pool Configuration:")
        print(f"  Snapshot every: {pool_snapshot_every:,} games")
        print(f"  Max size: {pool_max_size}")
        print(f"  Sample rates: Pool={pool_sample_rate:.0%}, Pubeval={pubeval_sample_rate:.0%}, Random={random_sample_rate:.0%}")

        if len(opponent_pool) > 0:
            print(f"\n{'='*60}")
            print("EXISTING OPPONENT POOL FOUND")
            print(f"{'='*60}")
            opponent_pool.verify_snapshots()
            print(f"{'='*60}\n")
        else:
            print(f"  Pool is empty - will seed with initial agent")
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

    # League
    league = CheckpointLeague(
        checkpoint_dir=checkpoint_base_path / f"league_{model_size}",
        agent_module_name="ppo_agent"
    )

    # Baseline
    baseline = pubeval if eval_vs == "pubeval" else randomAgent

    state = TrainingState(
        # Static / config
        model_size=model_size, device=device, eval_vs=eval_vs,
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval,
        use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every,
        pool_max_size=pool_max_size,
        pool_sample_rate=pool_sample_rate,
        pubeval_sample_rate=pubeval_sample_rate,
        random_sample_rate=random_sample_rate,
        use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k,
        use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every,
        n_eval_league=n_eval_league,

        # Runtime
        agent_instance=agent_instance,
        checkpoint_base_path=checkpoint_base_path,
        best_ckpt_path=best_ckpt_path,
        opponent_pool=opponent_pool,
        league=league,
        baseline=baseline,
        games_done=0,
        next_eval_at=n_epochs,
        next_snapshot_at=pool_snapshot_every
    )

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Opponent Pool: {'ENABLED' if state.use_opponent_pool else 'DISABLED'}")
    if state.use_opponent_pool and state.opponent_pool:
        print(f"  Initial pool size: {len(state.opponent_pool)}")
        print(f"  Pool ramp: 0% @ {state.pool_start_games:,} → {state.pool_target_rate*100:.0f}% @ {state.pool_ramp_end:,} games")
        print(f"  First snapshot at: {state.next_snapshot_at:,} games")
        print(f"  Snapshot frequency: every {state.pool_snapshot_every:,} games")
    print(f"Evaluation: every {state.n_epochs:,} games")
    print(f"Total games: {state.n_games:,}")
    print(f"{'='*60}\n")

    # Optional warm start
    if state.use_bc_warmstart:
        agent.warmstart_with_pubeval(batch_iter, epochs=3, assume_second_roll=False)

    return state


# =========================
# One training step
# =========================

def train_step(state: TrainingState, train_bar: tqdm):
    """Run one batched rollout of complete games and handle pool snapshots if thresholds are crossed."""
    ai = state.agent_instance

    # Effective pool rate (ramped)
    if state.use_opponent_pool and state.opponent_pool and len(state.opponent_pool) > 0:
        if state.games_done < state.pool_start_games:
            effective_pool_rate = 0.0
        elif state.games_done < state.pool_ramp_end:
            t = (state.games_done - state.pool_start_games) / (state.pool_ramp_end - state.pool_start_games)
            t = max(0.0, min(1.0, t))
            effective_pool_rate = state.pool_target_rate * t
        else:
            effective_pool_rate = state.pool_target_rate
    else:
        effective_pool_rate = 0.0

    # Opponent selection
    opponent = None
    opponent_type = 'self_play'
    r = random.random()
    if r < effective_pool_rate:
        bias_recent = state.games_done >= state.pool_ramp_end
        if random.random() < 0.5 and state.best_ckpt_path.exists():
            try:
                agent_mod = importlib.import_module("ppo_agent")
                opponent = agent_mod.PPOAgent(device=state.device)
                opponent.load(str(state.best_ckpt_path), map_location=state.device, load_optimizer=False)
                opponent.set_eval_mode(True)
                opponent_type = 'pool_best'
            except Exception:
                opponent = state.opponent_pool.sample_opponent(bias_recent=bias_recent)
                opponent_type = 'pool' if opponent is not None else 'self_play'
        else:
            opponent = state.opponent_pool.sample_opponent(bias_recent=bias_recent)
            opponent_type = 'pool' if opponent is not None else 'self_play'
    elif r < effective_pool_rate + state.pubeval_sample_rate:
        opponent = pubeval
        opponent_type = 'pubeval'
    elif r < effective_pool_rate + state.pubeval_sample_rate + state.random_sample_rate:
        opponent = randomAgent
        opponent_type = 'random'

    if opponent is None:
        opponent = ai
        opponent_type = 'self_play'

    # Play batch
    finished = play_games_batched(ai, opponent, batch_size=BATCH_SIZE, training=True)
    if state.games_done + finished > state.n_games:
        finished = state.n_games - state.games_done

    # Accounting
    state.games_done += finished
    train_bar.update(finished)
    state.opponent_stats[opponent_type] += finished

    # Progress bar postfix
    if state.games_done % 100 == 0 or state.games_done == state.n_games:
        postfix = {"steps": f"{ai.steps:,}", "upd": ai.updates}
        if hasattr(ai, 'rollout_stats'):
            stats = ai.rollout_stats
            if stats['policy_loss']:
                postfix["πL"] = f"{stats['policy_loss'][-1]:.3f}"
            if stats['value_loss']:
                postfix["VL"] = f"{stats['value_loss'][-1]:.3f}"
            if stats['grad_norm']:
                postfix["∇"] = f"{stats['grad_norm'][-1]:.2f}"

        total_opp_games = sum(state.opponent_stats.values())
        if total_opp_games > 0:
            postfix["self%"] = f"{100.0 * state.opponent_stats['self_play'] / total_opp_games:.0f}"
            postfix["pool%"] = f"{100.0 * state.opponent_stats['pool'] / total_opp_games:.0f}"
            postfix["pub%"]  = f"{100.0 * state.opponent_stats['pubeval'] / total_opp_games:.0f}"
            postfix["rnd%"]  = f"{100.0 * state.opponent_stats['random'] / total_opp_games:.0f}"
        train_bar.set_postfix(postfix)

    # Pool snapshot thresholds
    while state.use_opponent_pool and state.opponent_pool and state.games_done >= state.next_snapshot_at and state.next_snapshot_at > 0:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT CHECKPOINT at {state.games_done:,} games")
        print(f"{'='*60}")

        latest_path = state.checkpoint_base_path / f"latest_ppo_{state.model_size}.pt"
        ai.save(str(latest_path))

        if not latest_path.exists():
            print(f"❌ ERROR: Checkpoint not saved at {latest_path}")
            print(f"   Check that {latest_path.parent} is writable")
        else:
            print(f"✓ Checkpoint saved: {latest_path}")
            print(f"  File size: {latest_path.stat().st_size / 1024:.1f} KB")

        print(f"\n🔍 Checking competence before adding to pool...")
        ai.set_eval_mode(True)
        wr_vs_random = evaluate(ai, randomAgent, n_eval=40,
                                label="vs random (gate check)", debug_sides=False,
                                use_lookahead=False)
        ai.set_eval_mode(False)

        min_competence = 60.0
        if wr_vs_random >= min_competence:
            print(f"✓ Snapshot meets competence threshold ({wr_vs_random:.1f}% ≥ {min_competence}%)")
            state.opponent_pool.add_snapshot(latest_path, label=f"(after {state.next_snapshot_at:,} games, WR={wr_vs_random:.0f}%)")
            print(f"\n{state.opponent_pool.get_pool_info()}")
        else:
            print(f"❌ Snapshot rejected ({wr_vs_random:.1f}% < {min_competence}%)")
            print(f"   Not adding to pool to avoid polluting curriculum")

        if len(state.opponent_pool) > 0 and state.games_done == state.next_snapshot_at:
            print("\n🔍 Testing pool loading...")
            test_opp = state.opponent_pool.sample_opponent(bias_recent=True)
            if test_opp is not None:
                print(f"✓ Successfully loaded opponent from pool!")
                print(f"  Opponent steps: {getattr(test_opp, 'steps', 'N/A')}")
            else:
                print(f"❌ WARNING: Could not load opponent from pool")
                print(f"   Pool will not be used in training!")

        print(f"{'='*60}\n")
        state.next_snapshot_at += state.pool_snapshot_every


# =========================
# Evaluation / validation
# =========================

def validation_step(state: TrainingState):
    """Run one or more evaluation cycles if thresholds are reached."""
    ai = state.agent_instance
    ran_eval = False

    while state.games_done >= state.next_eval_at and state.next_eval_at > 0:
        ran_eval = True
        print()
        ai.set_eval_mode(True)
        print("[Eval] Agent in EVAL mode")
        print(f"\n--- Evaluation after {state.next_eval_at:,} games ---")

        wr = evaluate(ai, state.baseline, state.n_eval,
                      label=f"vs {state.eval_vs} (greedy)",
                      debug_sides=False, use_lookahead=False)
        state.perf_data['vs_baseline'].append(wr)

        if wr > state.best_wr:
            state.best_wr = wr
            ai.save(str(state.best_ckpt_path))
            print(f"  🌟 NEW BEST vs pubeval: {wr:.1f}% (saved to {state.best_ckpt_path.name})")

        if state.use_eval_lookahead:
            wr_lookahead = evaluate(ai, state.baseline, min(100, state.n_eval),
                                    label=f"vs {state.eval_vs}",
                                    debug_sides=False, use_lookahead=True,
                                    lookahead_k=state.eval_lookahead_k)
            state.perf_data['vs_baseline_lookahead'].append(wr_lookahead)
            print(f"  Lookahead improvement: +{(wr_lookahead - wr):.1f}% points")

        wr_rand = evaluate(ai, randomAgent, max(50, state.n_eval // 2),
                           label="vs random", debug_sides=False, use_lookahead=False)
        state.perf_data['vs_random'].append(wr_rand)
        if wr_rand < 50.0:
            print(f"  ⚠️  WARNING: Only {wr_rand:.1f}% vs random!")

        ai.set_eval_mode(False)
        state.next_eval_at += state.n_epochs

    return ran_eval


# =========================
# Orchestrator (public API)
# =========================

def train(
    n_games=200_000,
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
    n_eval_league=50,
    device='cpu'
):
    # 1) Initialize
    state = initialize_training(
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval, eval_vs=eval_vs,
        model_size=model_size, use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every, pool_max_size=pool_max_size,
        pool_sample_rate=pool_sample_rate, pubeval_sample_rate=pubeval_sample_rate,
        random_sample_rate=random_sample_rate, use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k, use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every, n_eval_league=n_eval_league,
        device=device
    )

    # 2) Training loop
    train_bar = tqdm(total=state.n_games, desc="Training", unit="game")
    try:
        while state.games_done < state.n_games:
            train_step(state, train_bar)
            validation_step(state)
    finally:
        train_bar.close()

    # 3) Final reporting
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Model size: {state.model_size.upper()}")
    print()
    print("Opponent Usage:")
    total_games = sum(state.opponent_stats.values())
    if total_games > 0:
        for opp_type, count in state.opponent_stats.items():
            pct = 100.0 * count / total_games
            print(f"  {opp_type.capitalize()}: {count:,} ({pct:.1f}%)")

    print()
    print(f"Final best win-rate vs {state.eval_vs}: {state.best_wr:.1f}%")
    if state.perf_data['vs_league_avg']:
        print(f"Final league average: {state.perf_data['vs_league_avg'][-1]:.1f}%")

    ai = state.agent_instance
    print()
    print("Final PPO State:")
    print(f"  Updates: {ai.updates}")
    print(f"  Steps: {ai.steps}")
    print(f"  Entropy coef: {ai.current_entropy_coef:.4f}")

    if hasattr(ai, 'rollout_stats'):
        stats = ai.rollout_stats
        if stats['nA_values']:
            print(f"  Avg legal actions: {np.mean(stats['nA_values'][-1000:]):.1f}")
        if stats['masked_entropy']:
            print(f"  Final entropy: {np.mean(stats['masked_entropy'][-10:]):.4f}")

    print("=" * 70)

    plot_perf(state.perf_data, title=f"PPO Training ({state.model_size.upper()} model)")



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
                       help='Quick test: small model, 10k games, frequent evals')

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
            n_games=10_000,
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
            use_bc_warmstart=False,  # Jump-start with pubeval imitation
            league_checkpoint_every=10_000,
            device='cpu',
        )
    else:
        train(
            n_games=args.n_games,
            n_epochs=args.n_epochs,
            n_eval=200,
            eval_vs="pubeval",
            model_size=args.model_size,
            use_opponent_pool=True,
            pool_snapshot_every=5_000,
            pool_max_size=12,
            pool_sample_rate=0.90,      # 50% pool (self-play curriculum)
            pubeval_sample_rate=0.10,   # 45% pubeval (strong baseline)
            random_sample_rate=0.00,    # 0% random (robustness only)
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            device = 'cpu',
        )
