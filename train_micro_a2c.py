#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Micro-A2C Training Script - Adapted from PPO training script

Key changes from PPO version:
1. Import agent_ac_adv_micro instead of ppo_agent
2. Remove PPO-specific config adjustments (ppo_epochs, entropy_min)
3. Update OpponentPool agent_module_name to "agent_ac_adv_micro"
4. Adjust print statements to reference A2C instead of PPO
5. Remove rollout buffer references (A2C updates immediately)
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
from datetime import datetime

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
import a2c_micro_agent as agent  # CHANGED: import micro-A2C agent
from opponent_pool import OpponentPool
from utils import _ensure_dir, _safe_save_agent, plot_perf, _is_empty_move, _apply_move_sequence, flip_to_pov_plus1, get_device
from play_games import play_games_batched
from evaluate import evaluate, CheckpointLeague

# Set seeds for reproducibility
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Local checkpoint directory
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
    start_step: int
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
    timestamp: str

    # Curriculum tuning
    pool_start_games: int = 5_000
    pool_ramp_end: int = 50_000
    pool_target_rate: float = 0.30

    # Mutable runtime state
    agent_instance: "agent.MicroA2CAgent" = None  # CHANGED: type annotation
    checkpoint_base_path: Path = None
    best_ckpt_path: Path = None
    latest_ckpt_path: Path = None
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
    device='cpu',
    resume=None
) -> TrainingState:
    print("\n" + "=" * 70)
    print("MICRO-A2C TRAINING")  # CHANGED: title
    print("=" * 70)
    print(f"Model size: {model_size.upper()}")
    print(f"Total games: {n_games:,}")
    print(f"Evaluate every: {n_epochs:,} games")
    print(f"Evaluation games: {n_eval}")
    print(f"Baseline: {eval_vs}")
    print(f"Lookahead: {use_eval_lookahead} (k={eval_lookahead_k})")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    # Agent initialization - NO PPO-specific config tweaks
    cfg = agent.get_config(model_size)
    agent_instance = agent.MicroA2CAgent(config=cfg, device=device)  # CHANGED: MicroA2CAgent
    
    if resume is not None:
       agent_instance.load(resume)

    start_step = agent_instance.steps

    # Checkpoint paths with time and date to avoid overwriting
    checkpoint_base_path = Path(CHECKPOINT_DIR)
    _ensure_dir(checkpoint_base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_micro_a2c_{model_size}_{timestamp}.pt"  # CHANGED: filename
    best_ckpt_path = checkpoint_base_path / f"best_so_far_{model_size}_{timestamp}.pt"
    latest_ckpt_path = checkpoint_base_path / f"latest_{model_size}_{timestamp}.pt"

    # Opponent pool (optional)
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{model_size}"
        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name="agent_ac_adv_micro",  # CHANGED: module name
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
            latest_path = agent.CHECKPOINT_PATH.parent / f"latest_micro_a2c_{model_size}.pt"  # CHANGED: filename
            agent_instance.save(str(latest_path))
            opponent_pool.add_snapshot(latest_path, label="(seed at 0 games)")
            print(f"✓ Seed snapshot added")
            print(f"{opponent_pool.get_pool_info()}")
            print(f"{'='*60}\n")
        print()

    # League
    league = CheckpointLeague(
        checkpoint_dir=checkpoint_base_path / f"league_{model_size}",
        agent_module_name="agent_ac_adv_micro"  # CHANGED: module name
    )

    # Baseline
    baseline = pubeval if eval_vs == "pubeval" else randomAgent

    state = TrainingState(
        # Static / config
        model_size=model_size, device=device, eval_vs=eval_vs,
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval,
        start_step=start_step, use_opponent_pool=use_opponent_pool,
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
        timestamp=timestamp,

        # Runtime
        agent_instance=agent_instance,
        checkpoint_base_path=checkpoint_base_path,
        best_ckpt_path=best_ckpt_path,
        latest_ckpt_path=latest_ckpt_path,
        opponent_pool=opponent_pool,
        league=league,
        baseline=baseline,
        games_done=0,
        next_eval_at=0,
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

    # Note: No BC warmstart for A2C (TD(λ) learning is different)

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
                agent_mod = importlib.import_module("agent_ac_adv_micro")  # CHANGED: module name
                opponent = agent_mod.MicroA2CAgent(device=state.device)  # CHANGED: class name
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
        if hasattr(ai, 'stats'):  # CHANGED: A2C uses 'stats' not 'rollout_stats'
            stats = ai.stats
            if stats['td_errors']:
                postfix["δ"] = f"{np.mean(stats['td_errors'][-100:]):.4f}"
            if stats['values']:
                postfix["V"] = f"{np.mean(stats['values'][-100:]):.3f}"
            if stats['entropy']:
                postfix["H"] = f"{np.mean(stats['entropy'][-100:]):.3f}"

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

        latest_path = state.checkpoint_base_path / f"latest_micro_a2c_{state.model_size}.pt"  # CHANGED: filename
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

    while state.games_done >= state.next_eval_at and state.next_eval_at >= 0:
        ran_eval = True
        print()
        ai.set_eval_mode(True)
        # Save latest checkpoint
        ai.save(str(state.latest_ckpt_path))
        print("[Eval] Agent in EVAL mode")
        print(f"\n--- Evaluation after {state.next_eval_at:,} games ---")

        wr = evaluate(ai, state.baseline, state.n_eval,
                      label=f"vs {state.eval_vs} (greedy)",
                      debug_sides=False, use_lookahead=False)
        state.perf_data['vs_baseline'].append(wr)

        if wr > state.best_wr:
            state.best_wr = wr
            ai.save(str(state.best_ckpt_path))
            print(f"  🌟 NEW BEST vs {state.eval_vs}: {wr:.1f}% (saved to {state.best_ckpt_path.name})")

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
    device='cpu',
    resume=None
):
    # Initialize
    state = initialize_training(
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval, eval_vs=eval_vs,
        model_size=model_size, use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every, pool_max_size=pool_max_size,
        pool_sample_rate=pool_sample_rate, pubeval_sample_rate=pubeval_sample_rate,
        random_sample_rate=random_sample_rate, use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k, use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every, n_eval_league=n_eval_league,
        device=device, resume=resume
    )

    # Validate initial model
    validation_step(state)

    # Training loop
    train_bar = tqdm(total=state.n_games, desc="Training", unit="game")
    try:
        while state.games_done < state.n_games:
            train_step(state, train_bar)
            validation_step(state)
    finally:
        train_bar.close()

    # Final reporting
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
    print("Final Micro-A2C State:")  # CHANGED: title
    print(f"  Updates: {ai.updates}")
    print(f"  Steps: {ai.steps}")
    print(f"  Entropy coef: {ai.current_entropy_coef:.4f}")

    if hasattr(ai, 'stats'):  # CHANGED: stats structure
        stats = ai.stats
        if stats['td_errors']:
            print(f"  Avg TD error: {np.mean(stats['td_errors'][-1000:]):.4f}")
        if stats['entropy']:
            print(f"  Final entropy: {np.mean(stats['entropy'][-100:]):.4f}")

    print("=" * 70)

    plot_perf(state.perf_data, 0, n_games, n_epochs, 
              title=f"Micro-A2C Training ({state.model_size.upper()} model)",  # CHANGED: title
              timestamp=state.timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Micro-A2C agent for backgammon')  # CHANGED: description
    parser.add_argument('--model-size', type=str, default='large',
                       choices=['small', 'medium', 'large'],
                       help='Model size: small (CPU), medium (M1/M2/T4 GPU), large (good GPU)')
    parser.add_argument('--n-games', type=int, default=200_000,
                       help='Total number of training games')
    parser.add_argument('--n-epochs', type=int, default=5_000,
                       help='Evaluate every N games')
    parser.add_argument('--cpu-test', action='store_true',
                       help='Quick test: small model, 10k games, frequent evals')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for training')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to a .pt checkpoint to resume training from')
    args = parser.parse_args()

    # CPU test mode: fast settings for testing
    if args.cpu_test:
        print("\n" + "=" * 70)
        print("CPU TEST MODE")
        print("=" * 70)
        print("  Model: small")
        print("  Games: 10,000")
        print("  Eval every: 2,500 games")
        print("  Eval games: 100")
        print("  Baseline: pubeval")
        print("  Lookahead: disabled")
        print("=" * 70 + "\n")

        train(
            n_games=10_000,
            n_epochs=2_500,
            n_eval=100,
            eval_vs="pubeval",
            model_size='small',
            use_opponent_pool=False,  # Start without pool
            pool_snapshot_every=5_000,
            pubeval_sample_rate=0.30,  # 30% pubeval exposure
            random_sample_rate=0.10,   # 10% random for robustness
            use_eval_lookahead=False,  # Faster testing
            eval_lookahead_k=3,
            use_bc_warmstart=False,
            league_checkpoint_every=10_000,
            device=args.device,
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
            pool_sample_rate=0.40,      # 40% pool (self-play curriculum)
            pubeval_sample_rate=0.30,   # 30% pubeval (strong baseline)
            random_sample_rate=0.30,    # 30% random (robustness)
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            device=args.device,
            resume=args.resume
        )
