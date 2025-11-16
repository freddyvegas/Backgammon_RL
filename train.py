#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Training Script for PPO / Micro-A2C

Supports:
- Algorithms: PPO, Micro-A2C
- Model sizes: small / medium / large
- Agent types for PPO: MLP / transformer
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
torch.backends.cudnn.benchmark = True
import argparse
import importlib
from dataclasses import dataclass, field
from datetime import datetime

import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util

import ppo_agent as ppo_agent
import ppo_transformer_agent as transformer_agent
import a2c_micro_agent as a2c_agent

from opponent_pool import OpponentPool
from utils import (
    _ensure_dir, _safe_save_agent, plot_perf,
    _is_empty_move, _apply_move_sequence,
    flip_to_pov_plus1, get_device
)
from play_games import play_games_batched, play_games_batched_transformer
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


# =========================
# Core training structures
# =========================

@dataclass
class TrainingState:
    # Config / static
    algo: str                 # "ppo" or "micro-a2c"
    model_size: str
    device: str
    eval_vs: str
    n_games: int
    n_epochs: int
    n_eval: int
    start_step: int
    batch_size: int
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
    agent_type: str           # "MLP" or "transformer" (PPO only, ignored for A2C)

    # Curriculum tuning
    pool_start_games: int = 5_000
    pool_ramp_end: int = 50_000
    pool_target_rate: float = 0.30

    # Mutable runtime state
    agent_instance: object = None
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
    algo="ppo",
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
    agent_type='MLP',      # PPO only
    resume=None,
    batch_size=8
) -> TrainingState:

    algo = algo.lower()
    if algo not in ("ppo", "micro-a2c"):
        raise ValueError(f"Unsupported algo '{algo}', use 'ppo' or 'micro-a2c'.")

    print("\n" + "=" * 70)
    print(f"{'PPO' if algo == 'ppo' else 'MICRO-A2C'} TRAINING")
    print("=" * 70)
    print(f"Algorithm: {algo}")
    print(f"Model size: {model_size.upper()}")
    print(f"Total games: {n_games:,}")
    print(f"Evaluate every: {n_epochs:,} games")
    print(f"Evaluation games: {n_eval}")
    print(f"Baseline: {eval_vs}")
    print(f"Lookahead: {use_eval_lookahead} (k={eval_lookahead_k})")
    print(f"Device: {device}")
    if algo == "ppo":
        print(f"Agent type: {agent_type}")
    print(f"Batch size: {batch_size}")
    print("=" * 70 + "\n")

    # -------- Agent instantiation --------
    if algo == "ppo":
        if agent_type == 'transformer':
            cfg = transformer_agent.get_config(model_size)
            agent_instance = transformer_agent.PPOAgent(config=cfg, device=device)
        else:
            cfg = ppo_agent.get_config(model_size)
            agent_instance = ppo_agent.PPOAgent(config=cfg, device=device)
    else:  # Micro-A2C
        cfg = a2c_agent.get_config(model_size)
        agent_instance = a2c_agent.MicroA2CAgent(config=cfg, device=device)

    # Resume if requested
    if resume is not None:
        # Path objects are usually fine, but to be safe, cast to str
        agent_instance.load(str(resume))

    start_step = agent_instance.steps

    # -------- Checkpoint paths --------
    checkpoint_base_path = Path(CHECKPOINT_DIR)
    _ensure_dir(checkpoint_base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use different prefixes for PPO vs A2C
    ckpt_tag = "ppo" if algo == "ppo" else "micro_a2c"

    if algo == "ppo":
        # Keep PPO's global CHECKPOINT_PATH behavior
        ppo_agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_{ckpt_tag}_{model_size}_{timestamp}.pt"
    else:
        a2c_agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_{ckpt_tag}_{model_size}_{timestamp}.pt"

    best_ckpt_path = checkpoint_base_path / f"best_so_far_{ckpt_tag}_{model_size}_{timestamp}.pt"
    latest_ckpt_path = checkpoint_base_path / f"latest_{ckpt_tag}_{model_size}_{timestamp}.pt"

    # -------- Opponent pool (optional) --------
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{ckpt_tag}_{model_size}"
        pool_agent_module = "ppo_agent" if algo == "ppo" else "a2c_micro_agent"

        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name=pool_agent_module,
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
            latest_path = checkpoint_base_path / f"latest_{ckpt_tag}_{model_size}.pt"
            agent_instance.save(str(latest_path))
            opponent_pool.add_snapshot(latest_path, label="(seed at 0 games)")
            print(f"✓ Seed snapshot added")
            print(f"{opponent_pool.get_pool_info()}")
            print(f"{'='*60}\n")
        print()

    # -------- League --------
    league = CheckpointLeague(
        checkpoint_dir=checkpoint_base_path / f"league_{ckpt_tag}_{model_size}",
        agent_module_name="ppo_agent" if algo == "ppo" else "a2c_micro_agent"
    )

    # -------- Baseline --------
    baseline = pubeval if eval_vs == "pubeval" else randomAgent

    state = TrainingState(
        # Static / config
        algo=algo,
        model_size=model_size,
        device=device,
        eval_vs=eval_vs,
        n_games=n_games,
        n_epochs=n_epochs,
        n_eval=n_eval,
        batch_size=batch_size,
        start_step=start_step,
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
        timestamp=timestamp,
        agent_type=agent_type,

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
    print(f"Algorithm: {state.algo}")
    print(f"Opponent Pool: {'ENABLED' if state.use_opponent_pool else 'DISABLED'}")
    if state.use_opponent_pool and state.opponent_pool:
        print(f"  Initial pool size: {len(state.opponent_pool)}")
        print(f"  Pool ramp: 0% @ {state.pool_start_games:,} → {state.pool_target_rate*100:.0f}% @ {state.pool_ramp_end:,} games")
        print(f"  First snapshot at: {state.next_snapshot_at:,} games")
        print(f"  Snapshot frequency: every {state.pool_snapshot_every:,} games")
    print(f"Evaluation: every {state.n_epochs:,} games")
    print(f"Total games: {state.n_games:,}")
    print(f"Batch size: {state.batch_size}")
    print(f"{'='*60}\n")

    if state.use_bc_warmstart:
        # In the original PPO script this was not wired correctly (batch_iter undefined),
        # so we just warn and skip for now.
        print("⚠️  use_bc_warmstart=True requested, but BC warmstart is not implemented in this unified script. Skipping.")

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
                if state.algo == "ppo":
                    agent_mod = importlib.import_module("ppo_agent")
                    OppClass = getattr(agent_mod, "PPOAgent")
                    opponent = OppClass(device=state.device)
                else:
                    agent_mod = importlib.import_module("a2c_micro_agent")
                    OppClass = getattr(agent_mod, "MicroA2CAgent")
                    opponent = OppClass(device=state.device)

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
    if state.algo == "ppo" and state.agent_type == 'transformer':
        finished = play_games_batched_transformer(ai, opponent, batch_size=state.batch_size, training=True)
    else:
        finished = play_games_batched(ai, opponent, batch_size=state.batch_size, training=True)

    if state.games_done + finished > state.n_games:
        finished = state.n_games - state.games_done

    # Accounting
    state.games_done += finished
    train_bar.update(finished)
    state.opponent_stats[opponent_type] += finished

    # Progress bar postfix
    if state.games_done % 100 == 0 or state.games_done == state.n_games:
        postfix = {"steps": f"{ai.steps:,}", "upd": ai.updates}

        if state.algo == "ppo":
            if hasattr(ai, 'rollout_stats') and isinstance(ai.rollout_stats, dict):
                stats = ai.rollout_stats
                pol_loss = stats.get('policy_loss', [])
                val_loss = stats.get('value_loss', [])
                grad_norm = stats.get('grad_norm', [])
                if pol_loss:
                    postfix["πL"] = f"{pol_loss[-1]:.3f}"
                if val_loss:
                    postfix["VL"] = f"{val_loss[-1]:.3f}"
                if grad_norm:
                    postfix["∇"] = f"{grad_norm[-1]:.2f}"
        else:  # Micro-A2C stats
            if hasattr(ai, 'stats') and isinstance(ai.stats, dict):
                stats = ai.stats
                td_errors = stats.get('td_errors', [])
                values = stats.get('values', [])
                entropy = stats.get('entropy', [])
                if td_errors:
                    postfix["δ"] = f"{np.mean(td_errors[-100:]):.4f}"
                if values:
                    postfix["V"] = f"{np.mean(values[-100:]):.3f}"
                if entropy:
                    postfix["H"] = f"{np.mean(entropy[-100:]):.3f}"

        total_opp_games = sum(state.opponent_stats.values())
        if total_opp_games > 0:
            postfix["self%"] = f"{100.0 * state.opponent_stats['self_play'] / total_opp_games:.0f}"
            postfix["pool%"] = f"{100.0 * state.opponent_stats['pool'] / total_opp_games:.0f}"
            postfix["pub%"]  = f"{100.0 * state.opponent_stats['pubeval'] / total_opp_games:.0f}"
            postfix["rnd%"]  = f"{100.0 * state.opponent_stats['random'] / total_opp_games:.0f}"

        train_bar.set_postfix(postfix)

    # Pool snapshot thresholds
    ckpt_tag = "ppo" if state.algo == "ppo" else "micro_a2c"
    while state.use_opponent_pool and state.opponent_pool and state.games_done >= state.next_snapshot_at and state.next_snapshot_at > 0:
        print(f"\n{'='*60}")
        print(f"SNAPSHOT CHECKPOINT at {state.games_done:,} games")
        print(f"{'='*60}")

        latest_path = state.checkpoint_base_path / f"latest_{ckpt_tag}_{state.model_size}.pt"
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
            state.opponent_pool.add_snapshot(
                latest_path,
                label=f"(after {state.next_snapshot_at:,} games, WR={wr_vs_random:.0f}%)"
            )
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
            wr_lookahead = evaluate(
                ai, state.baseline, min(100, state.n_eval),
                label=f"vs {state.eval_vs}",
                debug_sides=False, use_lookahead=True,
                lookahead_k=state.eval_lookahead_k
            )
            state.perf_data['vs_baseline_lookahead'].append(wr_lookahead)
            print(f"  Lookahead improvement: +{(wr_lookahead - wr):.1f}% points")

        wr_rand = evaluate(
            ai, randomAgent, max(50, state.n_eval // 2),
            label="vs random", debug_sides=False, use_lookahead=False
        )
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
    algo="ppo",
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
    agent_type='MLP',
    resume=None,
    batch_size=8
):
    # Initialize
    state = initialize_training(
        algo=algo,
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval, eval_vs=eval_vs,
        model_size=model_size, use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every, pool_max_size=pool_max_size,
        pool_sample_rate=pool_sample_rate, pubeval_sample_rate=pubeval_sample_rate,
        random_sample_rate=random_sample_rate, use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k, use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every, n_eval_league=n_eval_league,
        device=device, agent_type=agent_type, resume=resume, batch_size=batch_size
    )

    # Validate initial model
    validation_step(state)

    print(f"Training on {state.batch_size} games in parallel")

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
    print(f"Algorithm: {state.algo}")
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
    print(f"Final {state.algo.upper()} State:")
    print(f"  Updates: {ai.updates}")
    print(f"  Steps: {ai.steps}")
    print(f"  Entropy coef: {ai.current_entropy_coef:.4f}")

    if state.algo == "ppo":
        if hasattr(ai, 'rollout_stats') and isinstance(ai.rollout_stats, dict):
            stats = ai.rollout_stats
            nA_values = stats.get('nA_values', [])
            masked_entropy = stats.get('masked_entropy', [])
            if nA_values:
                print(f"  Avg legal actions: {np.mean(nA_values[-1000:]):.1f}")
            if masked_entropy:
                print(f"  Final entropy: {np.mean(masked_entropy[-10:]):.4f}")
    else:
        if hasattr(ai, 'stats') and isinstance(ai.stats, dict):
            stats = ai.stats
            td_errors = stats.get('td_errors', [])
            entropy = stats.get('entropy', [])
            if td_errors:
                print(f"  Avg TD error: {np.mean(td_errors[-1000:]):.4f}")
            if entropy:
                print(f"  Final entropy: {np.mean(entropy[-100:]):.4f}")

    print("=" * 70)

    title_prefix = "PPO" if state.algo == "ppo" else "Micro-A2C"
    plot_perf(
        state.perf_data, 0, n_games, n_epochs,
        title=f"{title_prefix} Training ({state.model_size.upper()} model)",
        timestamp=state.timestamp
    )


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified trainer for PPO / Micro-A2C agents for backgammon')
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['ppo', 'micro-a2c'],
                        help='Which algorithm to train')
    parser.add_argument('--model-size', type=str, default='large',
                        choices=['small', 'medium', 'large'],
                        help='Model size: small (CPU), medium (M1/M2/T4 GPU), large (good GPU)')
    parser.add_argument('--n-games', type=int, default=200_000,
                        help='Total number of training games')
    parser.add_argument('--n-epochs', type=int, default=5_000,
                        help='Evaluate every N games')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--cpu-test', action='store_true',
                        help='Quick test: fewer games, frequent evals')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device type to train on (e.g. cpu, cuda, mps)')
    parser.add_argument('--agent-type', type=str, default='MLP',
                        help='PPO agent type: MLP or transformer')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to a .pt checkpoint to resume training from')
    args = parser.parse_args()

    # CPU test mode: fast settings for debugging
    if args.cpu_test:
        print("\n" + "=" * 70)
        print("CPU TEST MODE")
        print("=" * 70)
        print(f"  Algo: {args.algo}")
        print("  Model: small")
        print("  Games: 10,000")
        print("  Eval every: 2,500 games")
        print("  Eval games: 100")
        print("  Baseline: pubeval")
        print("  Lookahead: disabled")
        print("=" * 70 + "\n")

        train(
            algo=args.algo,
            n_games=10_000,
            n_epochs=2_500,
            n_eval=100,
            eval_vs="pubeval",
            model_size='small',
            use_opponent_pool=False,   # start without pool
            pool_snapshot_every=5_000,
            pubeval_sample_rate=0.30,
            random_sample_rate=0.10,
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            use_bc_warmstart=False,
            league_checkpoint_every=10_000,
            device=args.device,
            agent_type=args.agent_type,
            batch_size=args.batch_size,
            resume=args.resume
        )
    else:
        train(
            algo=args.algo,
            n_games=args.n_games,
            n_epochs=args.n_epochs,
            n_eval=200,
            eval_vs="pubeval",
            model_size=args.model_size,
            use_opponent_pool=True,
            pool_snapshot_every=5_000,
            pool_max_size=12,
            pool_sample_rate=0.40,
            pubeval_sample_rate=0.10,
            random_sample_rate=0.00,
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            device=args.device,
            agent_type=args.agent_type,
            resume=args.resume,
            batch_size=args.batch_size
        )
