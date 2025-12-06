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
import copy
from collections import Counter, deque, OrderedDict
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
import gnu_backgammon_player as gnubg_player

import ppo_agent as ppo_agent
import ppo_transformer_agent as transformer_agent
import a2c_micro_agent as a2c_agent
import agent_td_lambda_baseline as baseline_agent

from opponent_pool import OpponentPool
from utils import (
    _ensure_dir, _safe_save_agent, plot_perf, plot_perf_multi, plot_elo_history,
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


def resolve_agent_module(algo: str, agent_type: str):
    """Return (module_name, class_name) for initializing opponents/checkpoints."""
    if algo == "ppo":
        module = "ppo_transformer_agent" if agent_type == "transformer" else "ppo_agent"
        cls_name = "PPOAgent"
    elif algo == "baseline-td":
        module = "agent_td_lambda_baseline"
        cls_name = "TDLambdaAgent"
    else:
        module = "a2c_micro_agent"
        cls_name = "MicroA2CAgent"
    return module, cls_name


# =========================
# ELO tracking utilities
# =========================

@dataclass
class OpponentProfile:
    opponent_id: str
    label: str
    source: str
    base_weight: float = 1.0
    snapshot_id: int = None
    slow: bool = False


class EloTracker:
    def __init__(self, default_rating: float = 1500.0, k_factor: float = 32.0):
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.ratings = {}
        self.game_counts = Counter()

    def ensure_player(self, player_id: str):
        if player_id not in self.ratings:
            self.ratings[player_id] = self.default_rating
            self.game_counts[player_id] = 0

    def get_rating(self, player_id: str) -> float:
        self.ensure_player(player_id)
        return self.ratings[player_id]

    def expected_score(self, player_a: str, player_b: str) -> float:
        self.ensure_player(player_a)
        self.ensure_player(player_b)
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def record_match(self, player_a: str, player_b: str, result: float):
        """Record a match where result=1 if A wins, 0 for B."""
        if result not in (0.0, 1.0):
            return
        exp_a = self.expected_score(player_a, player_b)
        exp_b = 1.0 - exp_a
        self.ratings[player_a] += self.k_factor * (result - exp_a)
        self.ratings[player_b] += self.k_factor * ((1.0 - result) - exp_b)
        self.game_counts[player_a] += 1
        self.game_counts[player_b] += 1

    def get_rankings(self):
        for pid, rating in self.ratings.items():
            yield pid, rating, self.game_counts.get(pid, 0)


# =========================
# Core training structures
# =========================

@dataclass
class TrainingState:
    # Config / static
    algo: str                 # "ppo" or "micro-a2c"
    model_size: str
    device: str
    teacher: str
    eval_vs: str
    n_games: int
    n_epochs: int
    n_eval: int
    start_step: int
    batch_size: int
    use_opponent_pool: bool
    pool_snapshot_every: int
    pool_max_size: int
    use_eval_lookahead: bool
    eval_lookahead_k: int
    use_bc_warmstart: bool
    league_checkpoint_every: int
    n_eval_league: int
    timestamp: str
    agent_type: str           # "MLP" or "transformer" (PPO only, ignored for A2C)
    agent_module_name: str
    agent_class_name: str
    ckpt_tag: str
    slow_opponent_batch: int = 1

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
    elo_tracker: EloTracker = None
    opponent_profiles: dict = field(default_factory=dict)
    current_agent_id: str = "agent_current"
    opponent_cache_ttl: int = 1_000
    cached_opponents: dict = field(default_factory=dict)
    recent_window: int = 1_000
    recent_opponents: deque = field(default_factory=deque)
    recent_counts: Counter = field(default_factory=Counter)
    opponent_ctor_kwargs: dict = field(default_factory=dict)
    opponent_config_template: object = None
    elo_history: list = field(default_factory=list)
    perf_data: dict = field(default_factory=lambda: {
    'vs_baseline': [],
    'vs_baseline_lookahead': [],
    'vs_random': [],
    'vs_gnubg': [],
    'vs_league_avg': [],
    'vs_latest_checkpoint': []
})
    opponent_stats: Counter = field(default_factory=Counter)


def register_opponent_profile(state: TrainingState, opponent_id: str, label: str,
                              source: str, base_weight: float = 1.0,
                              snapshot_id: int = None, slow: bool = False):
    if opponent_id in state.opponent_profiles:
        profile = state.opponent_profiles[opponent_id]
        if snapshot_id is not None:
            profile.snapshot_id = snapshot_id
        return profile
    profile = OpponentProfile(
        opponent_id=opponent_id,
        label=label,
        source=source,
        base_weight=base_weight,
        snapshot_id=snapshot_id,
        slow=slow
    )
    state.opponent_profiles[opponent_id] = profile
    if state.elo_tracker:
        state.elo_tracker.ensure_player(opponent_id)
    return profile


def sync_pool_profiles(state: TrainingState):
    if not state.use_opponent_pool or not state.opponent_pool:
        return
    for _, snapshot_id in state.opponent_pool.snapshots:
        opp_id = f"pool_{snapshot_id}"
        register_opponent_profile(
            state,
            opponent_id=opp_id,
            label=f"Pool #{snapshot_id}",
            source='pool',
            base_weight=1.0,
            snapshot_id=snapshot_id
        )


def get_cached_opponent(state: TrainingState, opponent_id: str):
    entry = state.cached_opponents.get(opponent_id)
    if entry:
        return entry['agent']
    return None


def cache_opponent_instance(state: TrainingState, opponent_id: str, opponent):
    if opponent_id is None or opponent is None:
        return opponent
    state.cached_opponents[opponent_id] = {
        'agent': opponent,
        'last_used': state.games_done
    }
    return opponent


def touch_cached_opponent(state: TrainingState, opponent_id: str):
    entry = state.cached_opponents.get(opponent_id)
    if entry:
        entry['last_used'] = state.games_done


def prune_cached_opponents(state: TrainingState):
    ttl = getattr(state, 'opponent_cache_ttl', 0)
    if ttl <= 0:
        return
    stale_ids = []
    for opponent_id, entry in state.cached_opponents.items():
        if state.games_done - entry.get('last_used', 0) >= ttl:
            stale_ids.append(opponent_id)
    for opponent_id in stale_ids:
        state.cached_opponents.pop(opponent_id, None)


def record_recent_usage(state: TrainingState, opponent_type: str, count: int):
    if count <= 0 or state.recent_window <= 0:
        return
    for _ in range(count):
        while len(state.recent_opponents) >= state.recent_window:
            removed = state.recent_opponents.popleft()
            state.recent_counts[removed] -= 1
            if state.recent_counts[removed] <= 0:
                del state.recent_counts[removed]
        state.recent_opponents.append(opponent_type)
        state.recent_counts[opponent_type] = state.recent_counts.get(opponent_type, 0) + 1


def compute_challenge_weight(state: TrainingState, opponent_id: str,
                              min_weight: float = 0.05) -> float:
    if opponent_id == state.current_agent_id:
        return 0.0
    profile = state.opponent_profiles.get(opponent_id)
    if profile is None:
        return 0.0
    agent_rating = state.elo_tracker.get_rating(state.current_agent_id)
    opp_rating = state.elo_tracker.get_rating(opponent_id)
    expected = 1.0 / (1.0 + 10 ** ((opp_rating - agent_rating) / 400.0))
    closeness = 1.0 - (abs(expected - 0.5) * 2.0)
    closeness = max(min_weight, closeness)
    weight = closeness * profile.base_weight
    if profile.source == 'pool':
        weight *= compute_pool_weight_multiplier(state)
    if profile.slow:
        weight *= 0.5
    return weight


def print_elo_standings(state: TrainingState, max_entries: int = 10):
    if not state.elo_tracker:
        return
    rankings = sorted(state.elo_tracker.get_rankings(), key=lambda item: item[1], reverse=True)
    if not rankings:
        return
    print(f"{'-'*70}")
    print("ELO RATINGS")
    print(f"{'-'*70}")
    count = 0
    for player_id, rating, games in rankings:
        profile = state.opponent_profiles.get(player_id)
        label = profile.label if profile else player_id
        print(f"  {label:<24}  {rating:7.1f}  ({games} games)")
        count += 1
        if count >= max_entries:
            break
    print(f"{'-'*70}")


def compute_pool_weight_multiplier(state: TrainingState) -> float:
    if not state.use_opponent_pool or not state.opponent_pool:
        return 0.0
    if state.games_done < state.pool_start_games:
        return 0.0
    if state.games_done >= state.pool_ramp_end:
        return max(0.0, min(1.0, state.pool_target_rate))
    ramp = (state.games_done - state.pool_start_games) / max(1, (state.pool_ramp_end - state.pool_start_games))
    scaled_target = max(0.0, min(1.0, state.pool_target_rate))
    return max(0.0, min(1.0, ramp)) * scaled_target


def _pool_snapshot_exists(state: TrainingState, snapshot_id: int) -> bool:
    if not state.opponent_pool:
        return False
    return any(sid == snapshot_id for _, sid in state.opponent_pool.snapshots)


def get_available_opponent_ids(state: TrainingState):
    available = []
    for opponent_id, profile in state.opponent_profiles.items():
        if profile.source == 'agent':
            continue
        if profile.source == 'pool':
            if compute_pool_weight_multiplier(state) <= 0.0:
                continue
            if profile.snapshot_id is None or not _pool_snapshot_exists(state, profile.snapshot_id):
                continue
        elif profile.source == 'pool_best':
            if not state.best_ckpt_path.exists():
                continue
        elif profile.source == 'self_play':
            pass
        # reference opponents always available
        available.append(opponent_id)
    return available


def choose_opponent_profile(state: TrainingState):
    candidate_ids = get_available_opponent_ids(state)
    filtered = [cid for cid in candidate_ids if state.opponent_profiles.get(cid)]
    if not filtered:
        return state.opponent_profiles.get('self_play')
    weights = [compute_challenge_weight(state, cid) for cid in filtered]
    total_weight = sum(weights)
    if total_weight <= 0:
        return state.opponent_profiles.get('self_play')
    r = random.random() * total_weight
    cumulative = 0.0
    for cid, w in zip(filtered, weights):
        cumulative += w
        if r <= cumulative:
            return state.opponent_profiles[cid]
    return state.opponent_profiles[filtered[-1]]


def instantiate_opponent(state: TrainingState, profile: OpponentProfile):
    if profile is None:
        return state.agent_instance
    cached = get_cached_opponent(state, profile.opponent_id)
    if cached:
        return cached
    if profile.source == 'self_play':
        return state.agent_instance
    if profile.source == 'pubeval':
        return pubeval
    if profile.source == 'random':
        return randomAgent
    if profile.source == 'gnubg':
        return gnubg_player
    if profile.source == 'pool_best':
        if not state.best_ckpt_path.exists():
            return None
        try:
            agent_mod = importlib.import_module(state.agent_module_name)
            OppClass = getattr(agent_mod, state.agent_class_name)
            kwargs = dict(state.opponent_ctor_kwargs or {})
            if state.opponent_config_template is not None:
                kwargs['config'] = copy.deepcopy(state.opponent_config_template)
            opponent = OppClass(**kwargs)
            opponent.load(str(state.best_ckpt_path), map_location=state.device, load_optimizer=False)
            opponent.set_eval_mode(True)
            opponent._opponent_id = profile.opponent_id
            return cache_opponent_instance(state, profile.opponent_id, opponent)
        except Exception as exc:
            print(f"[Opponent] Failed to load best checkpoint opponent: {exc}")
            return None
    if profile.source == 'pool' and profile.snapshot_id is not None and state.opponent_pool:
        try:
            opponent = state.opponent_pool.load_snapshot(profile.snapshot_id)
            if opponent:
                opponent._opponent_id = profile.opponent_id
                return cache_opponent_instance(state, profile.opponent_id, opponent)
            return None
        except Exception as exc:
            print(f"[Opponent] Failed to load pool snapshot {profile.snapshot_id}: {exc}")
            return None
    return None


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
    use_eval_lookahead=True,
    eval_lookahead_k=3,
    use_bc_warmstart=False,
    league_checkpoint_every=20_000,
    n_eval_league=50,
    device='cpu',
    agent_type='MLP',
    resume=None,
    batch_size=8,
    teacher='pubeval'
) -> TrainingState:

    algo = algo.lower()
    if algo not in ("ppo", "micro-a2c", "baseline-td"):
        raise ValueError(f"Unsupported algo '{algo}', use 'ppo' or 'micro-a2c', or 'baseline-td'.")
    agent_module_name, agent_class_name = resolve_agent_module(algo, agent_type)

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
    teacher_module = None
    teacher = (teacher or 'pubeval').lower()
    if algo == 'ppo':
        if teacher == 'pubeval':
            teacher_module = pubeval
        elif teacher == 'gnubg':
            teacher_module = gnubg_player
        elif teacher in ('none', 'off', 'disabled'):
            teacher = 'none'
        else:
            raise ValueError(f"Unsupported teacher '{teacher}'. Use 'pubeval', 'gnubg', or 'none'.")
        print(f"Teacher: {teacher}")
    else:
        teacher = 'none'
    print(f"Batch size: {batch_size}")
    print("=" * 70 + "\n")

    opponent_config_template = None
    opponent_ctor_kwargs = {'device': device}

    # -------- Agent instantiation --------
    if algo == "ppo":
        if agent_type == 'transformer':
            cfg = transformer_agent.get_config(model_size)
            agent_instance = transformer_agent.PPOAgent(
                config=cfg,
                device=device,
                teacher_mode=teacher,
                teacher_module=teacher_module
            )
        else:
            cfg = ppo_agent.get_config(model_size)
            agent_instance = ppo_agent.PPOAgent(
                config=cfg,
                device=device,
                teacher_mode=teacher,
                teacher_module=teacher_module
            )
        if hasattr(agent_instance, 'set_training_horizon'):
            agent_instance.set_training_horizon(n_games)
        opponent_config_template = copy.deepcopy(cfg)
        opponent_ctor_kwargs.update({'teacher_mode': 'none', 'teacher_module': None})
    elif algo == "baseline-td":  # NEW: Add baseline option
        import agent_td_lambda_baseline as baseline_agent
        cfg = baseline_agent.get_config(model_size)
        agent_instance = baseline_agent.TDLambdaAgent(config=cfg, device=device)
        opponent_config_template = copy.deepcopy(cfg)
    else:  # Micro-A2C
        cfg = a2c_agent.get_config(model_size)
        agent_instance = a2c_agent.MicroA2CAgent(config=cfg, device=device)
        opponent_config_template = copy.deepcopy(cfg)
    start_step = agent_instance.steps

    # -------- Checkpoint paths --------
    checkpoint_base_path = Path(CHECKPOINT_DIR)
    _ensure_dir(checkpoint_base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Use different prefixes for PPO vs A2C vs Baseline
    if algo == "ppo":
        ckpt_tag = "ppo_transformer" if agent_type.lower() == "transformer" else "ppo"
    elif algo == "baseline-td":
        ckpt_tag = "baseline_td"
    else:
        ckpt_tag = "micro_a2c"

    if algo == "ppo":
        # Keep PPO's global CHECKPOINT_PATH behavior
        ppo_module = transformer_agent if agent_type == 'transformer' else ppo_agent
        ppo_module.CHECKPOINT_PATH = checkpoint_base_path / f"best_{ckpt_tag}_{model_size}_{timestamp}.pt"
    elif algo == "baseline-td":
        baseline_agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_{ckpt_tag}_{model_size}_{timestamp}.pt"
    else:
        a2c_agent.CHECKPOINT_PATH = checkpoint_base_path / f"best_{ckpt_tag}_{model_size}_{timestamp}.pt"

    best_ckpt_path = checkpoint_base_path / f"best_so_far_{ckpt_tag}_{model_size}_{timestamp}.pt"
    latest_ckpt_path = checkpoint_base_path / f"latest_{ckpt_tag}_{model_size}_{timestamp}.pt"

    # -------- Opponent pool (optional) --------
    opponent_pool = None
    if use_opponent_pool:
        pool_dir = checkpoint_base_path / f"opponent_pool_{ckpt_tag}_{model_size}"

        opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            agent_module_name=agent_module_name,
            max_size=pool_max_size,
            seed=RANDOM_SEED,
            device=device,
            ctor_kwargs=opponent_ctor_kwargs,
            config_template=opponent_config_template
        )
        print(f"\nOpponent Pool Configuration:")
        print(f"  Snapshot every: {pool_snapshot_every:,} games")
        print(f"  Max size: {pool_max_size}")

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
        agent_module_name=agent_module_name
    )

    # -------- Baseline --------
    if eval_vs == "pubeval":
        baseline = pubeval
    elif eval_vs == "gnubg":
        baseline = gnubg_player
    elif eval_vs == "random":
        baseline = randomAgent
    else:
        baseline = gnubg_player

    slow_batch = max(1, batch_size // 4)

    state = TrainingState(
        # Static / config
        algo=algo,
        model_size=model_size,
        device=device,
        teacher=teacher,
        eval_vs=eval_vs,
        n_games=n_games,
        n_epochs=n_epochs,
        n_eval=n_eval,
        batch_size=batch_size,
        start_step=start_step,
        use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every,
        pool_max_size=pool_max_size,
        use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k,
        use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every,
        n_eval_league=n_eval_league,
        timestamp=timestamp,
        agent_type=agent_type,
        agent_module_name=agent_module_name,
        agent_class_name=agent_class_name,
        ckpt_tag=ckpt_tag,
        slow_opponent_batch=slow_batch,

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
        next_snapshot_at=pool_snapshot_every,
        opponent_ctor_kwargs=opponent_ctor_kwargs,
        opponent_config_template=opponent_config_template
    )

    state.elo_tracker = EloTracker()
    state.elo_tracker.ensure_player(state.current_agent_id)
    register_opponent_profile(state, state.current_agent_id, "Current Agent", "agent", base_weight=0.0)
    register_opponent_profile(state, "self_play", "Self-Play", "self_play", base_weight=0.2)
    register_opponent_profile(state, "pubeval", "Pubeval", "pubeval")
    register_opponent_profile(state, "random", "Random", "random")
    register_opponent_profile(state, "gnubg", "GNU Backgammon", "gnubg", slow=True)
    register_opponent_profile(state, "best_checkpoint", "Best Checkpoint", "pool_best")
    sync_pool_profiles(state)
    state.elo_history.append((0, state.elo_tracker.get_rating(state.current_agent_id)))

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
    if state.algo == 'ppo':
        print(f"Teacher: {state.teacher}")
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
def play_single_game(agent_obj, opponent, training=True, result_callback=None):
    """Play a single game for non-batched agents."""
    board = backgammon.init_board()
    player = 1
    
    if hasattr(agent_obj, 'episode_start'):
        agent_obj.episode_start()
    if hasattr(opponent, 'episode_start'):
        opponent.episode_start()
    
    while not backgammon.game_over(board):
        dice = backgammon.roll_dice()
        n_moves = 2 if dice[0] == dice[1] else 1
        
        for i in range(n_moves):
            if backgammon.game_over(board):
                break
            
            if player == 1:
                move = agent_obj.action(board.copy(), dice, player, i, train=training)
            else:
                if hasattr(opponent, 'action'):
                    move = opponent.action(board.copy(), dice, player, i, train=False)
                else:
                    # Module-style opponent (pubeval, random)
                    board_flipped = flipped_util.flip_board(board.copy())
                    move = opponent.action(board_flipped, dice, 1, i)
                    if len(move) > 0:
                        move = flipped_util.flip_move(move)
            
            if len(move) > 0:
                board = backgammon.update_board(board, move, player)
            
            if backgammon.game_over(board):
                break
        
        player = -player
    
    winner = 1 if board[27] == 15 else -1
    if callable(result_callback):
        result_callback(1 if winner == 1 else -1)
    
    if hasattr(agent_obj, 'end_episode'):
        agent_obj.end_episode(winner, board, perspective=1)
    if hasattr(opponent, 'end_episode'):
        opponent.end_episode(winner, board, perspective=-1)
    
    return winner

def train_step(state: TrainingState, train_bar: tqdm):
    """Run one batched rollout of complete games and handle pool snapshots if thresholds are crossed."""
    ai = state.agent_instance
    prune_cached_opponents(state)

    opponent_profile = choose_opponent_profile(state)
    opponent = instantiate_opponent(state, opponent_profile)
    opponent_type = opponent_profile.source if opponent_profile else 'self_play'
    opponent_id = opponent_profile.opponent_id if opponent_profile else 'self_play'

    if opponent is None:
        opponent = ai
        opponent_type = 'self_play'
        opponent_id = 'self_play'

    touch_cached_opponent(state, opponent_id)

    def make_result_callback(opponent_identifier: str):
        if opponent_identifier in (None, 'self_play', state.current_agent_id):
            return None

        def _cb(outcome: int):
            if outcome > 0:
                state.elo_tracker.record_match(state.current_agent_id, opponent_identifier, 1.0)
            elif outcome < 0:
                state.elo_tracker.record_match(state.current_agent_id, opponent_identifier, 0.0)
        return _cb

    result_callback = make_result_callback(opponent_id)

    train_metadata = {'opponent_type': opponent_type}
    if result_callback:
        train_metadata['result_callback'] = result_callback

    current_batch = state.slow_opponent_batch if (opponent_profile and opponent_profile.slow) else state.batch_size
    current_batch = max(1, min(current_batch, state.n_games - state.games_done))

    # Play batch or sequential
    if state.algo == "baseline-td":
        # Baseline TD-lambda doesn't support batching - play sequentially
        finished = 0
        for _ in range(current_batch):
            winner = play_single_game(
                ai, opponent, training=True,
                result_callback=result_callback
            )
            finished += 1
            if state.games_done + finished >= state.n_games:
                break
    elif state.algo == "ppo" and state.agent_type == 'transformer':
        finished = play_games_batched_transformer(
            ai, opponent, batch_size=current_batch, training=True, train_config=train_metadata
        )
    else:
        finished = play_games_batched(
            ai, opponent, batch_size=current_batch, training=True, train_config=train_metadata
        )

    if state.games_done + finished > state.n_games:
        finished = state.n_games - state.games_done

    # Accounting
    state.games_done += finished
    if hasattr(ai, 'update_lr_schedule') and state.algo == 'ppo':
        ai.update_lr_schedule(state.games_done)
    train_bar.update(finished)
    state.opponent_stats[opponent_type] += finished
    record_recent_usage(state, opponent_type, finished)
    touch_cached_opponent(state, opponent_id)

    postfix_items = [
        ("steps", f"{ai.steps:,}"),
        ("upd", ai.updates)
    ]

    if state.algo == "ppo":
        if hasattr(ai, 'rollout_stats') and isinstance(ai.rollout_stats, dict):
            stats = ai.rollout_stats
            pol_loss = stats.get('policy_loss', [])
            val_loss = stats.get('value_loss', [])
            grad_norm = stats.get('grad_norm', [])
            if pol_loss:
                postfix_items.append(("πL", f"{pol_loss[-1]:.3f}"))
            if val_loss:
                postfix_items.append(("VL", f"{val_loss[-1]:.3f}"))
            if grad_norm:
                postfix_items.append(("∇", f"{grad_norm[-1]:.2f}"))
    else:  # Micro-A2C or baseline-td stats
        if hasattr(ai, 'stats') and isinstance(ai.stats, dict):
            stats = ai.stats
            td_errors = stats.get('td_errors', [])
            values = stats.get('values', [])
            entropy = stats.get('entropy', [])
            if td_errors:
                postfix_items.append(("δ", f"{np.mean(td_errors[-100:]):.4f}"))
            if values:
                postfix_items.append(("V", f"{np.mean(values[-100:]):.3f}"))
            if entropy:
                postfix_items.append(("H", f"{np.mean(entropy[-100:]):.3f}"))

    recent_total = len(state.recent_opponents)
    counts = state.recent_counts if recent_total > 0 else state.opponent_stats
    denom = recent_total if recent_total > 0 else sum(state.opponent_stats.values())
    if denom > 0:
        postfix_items.append(("self%", f"{100.0 * counts.get('self_play', 0) / denom:.0f}"))
        postfix_items.append(("pool%", f"{100.0 * counts.get('pool', 0) / denom:.0f}"))
        postfix_items.append(("best%", f"{100.0 * counts.get('pool_best', 0) / denom:.0f}"))
        postfix_items.append(("gnu%", f"{100.0 * counts.get('gnubg', 0) / denom:.0f}"))
        postfix_items.append(("pub%", f"{100.0 * counts.get('pubeval', 0) / denom:.0f}"))
        postfix_items.append(("rnd%", f"{100.0 * counts.get('random', 0) / denom:.0f}"))
        postfix_items.append(("freqN", recent_total if recent_total > 0 else denom))
        postfix_items.append(("freqSrc", "recent" if recent_total > 0 else "total"))

    train_bar.set_postfix(OrderedDict(postfix_items))

    # Pool snapshot thresholds
    ckpt_tag = state.ckpt_tag

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
            new_id = state.opponent_pool.add_snapshot(
                latest_path,
                label=f"(after {state.next_snapshot_at:,} games, WR={wr_vs_random:.0f}%)"
            )
            if new_id is not None:
                register_opponent_profile(
                    state,
                    opponent_id=f"pool_{new_id}",
                    label=f"Pool #{new_id}",
                    source='pool',
                    snapshot_id=new_id
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
    """Run comprehensive evaluation against multiple opponents."""
    ai = state.agent_instance
    ran_eval = False

    while state.games_done >= state.next_eval_at and state.next_eval_at >= 0:
        ran_eval = True
        print()
        ai.set_eval_mode(True)
        
        # Save latest checkpoint
        ai.save(str(state.latest_ckpt_path))
        print("[Eval] Agent in EVAL mode")
        print(f"\n{'='*70}")
        print(f"EVALUATION AFTER {state.next_eval_at:,} GAMES")
        print(f"{'='*70}")
        
        # Evaluate vs GNU Backgammon FIRST (primary metric)
        try:
            wr_gnubg = evaluate(ai, gnubg_player, state.n_eval,  # Use full n_eval for primary metric
                               label="vs GNU Backgammon",
                               debug_sides=False, use_lookahead=False)
            if 'vs_gnubg' not in state.perf_data:
                state.perf_data['vs_gnubg'] = []
            state.perf_data['vs_gnubg'].append(wr_gnubg)
        except Exception as e:
            print(f"  ⚠️  GNU BG evaluation failed: {e}")
            wr_gnubg = 0.0
            if 'vs_gnubg' not in state.perf_data:
                state.perf_data['vs_gnubg'] = []
            state.perf_data['vs_gnubg'].append(0.0)
        
        # Evaluate vs Pubeval (secondary)
        wr_pubeval = evaluate(ai, pubeval, max(50, state.n_eval // 2),
                             label="vs Pubeval",
                             debug_sides=False, use_lookahead=False)
        state.perf_data['vs_baseline'].append(wr_pubeval)
        
        # Evaluate vs Random (sanity check)
        wr_random = evaluate(ai, randomAgent, max(50, state.n_eval // 2),
                            label="vs Random",
                            debug_sides=False, use_lookahead=False)
        state.perf_data['vs_random'].append(wr_random)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"  vs GNU BG:        {wr_gnubg:5.1f}%  ⭐ PRIMARY")
        print(f"  vs Pubeval:       {wr_pubeval:5.1f}%")
        print(f"  vs Random:        {wr_random:5.1f}%")
        print(f"{'='*70}\n")
        print_elo_standings(state)
        state.elo_history.append((state.games_done, state.elo_tracker.get_rating(state.current_agent_id)))
        
        # Check if new best (use GNU BG as primary metric)
        if wr_gnubg > state.best_wr:
            state.best_wr = wr_gnubg
            ai.save(str(state.best_ckpt_path))
            state.cached_opponents.pop('best_checkpoint', None)
            print(f"  🌟 NEW BEST vs GNU BG: {wr_gnubg:.1f}% (saved to {state.best_ckpt_path.name})")
        
        # Warning checks
        if wr_random < 60.0:
            print(f"  ⚠️  WARNING: Only {wr_random:.1f}% vs random!")
        
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
    use_eval_lookahead=True,
    eval_lookahead_k=3,
    use_bc_warmstart=False,
    league_checkpoint_every=20_000,
    n_eval_league=50,
    device='cpu',
    agent_type='MLP',
    resume=None,
    batch_size=8,
    teacher='pubeval'
):
    # Initialize
    state = initialize_training(
        algo=algo,
        n_games=n_games, n_epochs=n_epochs, n_eval=n_eval, eval_vs=eval_vs,
        model_size=model_size, use_opponent_pool=use_opponent_pool,
        pool_snapshot_every=pool_snapshot_every, pool_max_size=pool_max_size,
        use_eval_lookahead=use_eval_lookahead,
        eval_lookahead_k=eval_lookahead_k, use_bc_warmstart=use_bc_warmstart,
        league_checkpoint_every=league_checkpoint_every, n_eval_league=n_eval_league,
        device=device, agent_type=agent_type, resume=resume, batch_size=batch_size,
        teacher=teacher
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
            marker = " ⭐" if opp_type == "gnubg" else ""
            print(f"  {opp_type.capitalize()}: {count:,} ({pct:.1f}%){marker}")

    print()
    print(f"Final Performance Summary:")
    print(f"  vs GNU BG:   {state.perf_data['vs_gnubg'][-1]:.1f}%  ⭐ PRIMARY")
    print(f"  vs Pubeval:  {state.perf_data['vs_baseline'][-1]:.1f}%")
    print(f"  vs Random:   {state.perf_data['vs_random'][-1]:.1f}%")

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

    # Determine title based on algorithm
    if state.algo == "ppo":
        title_prefix = "PPO"
    elif state.algo == "baseline-td":
        title_prefix = "TD(λ) After-State Baseline"
    else:
        title_prefix = "Micro-A2C"

    # Plot with all three opponents
    plot_perf_multi(
        state.perf_data, 0, n_games, n_epochs,
        title=f"{title_prefix} Training ({state.model_size.upper()} model)",
        timestamp=state.timestamp
    )
    plot_elo_history(
        state.elo_history,
        n_games=n_games,
        title=f"ELO Progress ({state.model_size.upper()} model)",
        timestamp=state.timestamp
    )


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified trainer for PPO / Micro-A2C agents for backgammon')
    parser.add_argument('--algo', type=str, default='ppo',
                    choices=['ppo', 'micro-a2c', 'baseline-td'],
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
    parser.add_argument('--teacher', type=str, default='pubeval',
                        choices=['pubeval', 'gnubg', 'none'],
                        help='Teacher used for PPO DAGGER supervision')
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
        print("  Baseline: gnubg")
        print("  Lookahead: disabled")
        print("=" * 70 + "\n")

        train(
            algo=args.algo,
            n_games=10_000,
            n_epochs=2_500,
            n_eval=100,
            eval_vs="gnubg",
            model_size='small',
            use_opponent_pool=False,   # start without pool
            pool_snapshot_every=5_000,
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            use_bc_warmstart=False,
            league_checkpoint_every=10_000,
            device=args.device,
            agent_type=args.agent_type,
            batch_size=args.batch_size,
            resume=args.resume,
            teacher=args.teacher
        )
    else:
        train(
            algo=args.algo,
            n_games=args.n_games,
            n_epochs=args.n_epochs,
            n_eval=200,
            eval_vs="gnubg",
            model_size=args.model_size,
            use_opponent_pool=True,
            pool_snapshot_every=5_000,
            pool_max_size=12,
            use_eval_lookahead=False,
            eval_lookahead_k=3,
            device=args.device,
            agent_type=args.agent_type,
            resume=args.resume,
            batch_size=args.batch_size,
            teacher=args.teacher
        )
