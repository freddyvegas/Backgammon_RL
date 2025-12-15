# Backgammon_RL

Reinforcement-learning playground for modern Backgammon agents. The repository
contains a unified trainer, a suite of PPO architectures (MLP, CNN, and
Transformer), handcrafted opponents such as Tesauro's **pubeval** heuristic, and
wrappers around the official **GNU Backgammon** engine for benchmarking.

## Highlights
- **Unified training loop** (`train.py`) that can run PPO, Micro-A2C, or the TD(λ)
  baseline, manage checkpoints, and plot win-rate/ELO curves.
- **ELO-driven opponent sampling** that keeps training games competitive by
  weighting self-play, reference bots, and opponent-pool snapshots based on
  their current rating gap to the learner.
- **Multiple PPO backbones** (`ppo_agent.py`, `ppo_conv_agent.py`,
  `ppo_transformer_agent.py`) that share rollouts/teachers but differ in how
  they encode Backgammon states.
- **External experts for supervision/evaluation** including
  `pubeval_player.py` (Numba version of Tesauro's heuristic) and
  `gnu_backgammon_player.py` (bridge to the gnubg neural nets).

## Training with `train.py`

The trainer auto-detects CUDA, handles opponent pools, and periodically
evaluates the learner. Key flags:

```
usage: python train.py [--algo {ppo,micro-a2c,baseline-td}] \
                       [--agent-type {mlp,cnn,transformer}] \
                       [--model-size {small,medium,large}] \
                       [--device cpu|cuda|mps] \
                       [--teacher {pubeval,gnubg,none}] \
                       [--resume checkpoints/...pt] \
                       [--use-eval-lookahead --eval-lookahead-k 1] \
                       [--cpu-test]
```

Typical workflows:

1. **Full PPO run on GPU (default opponent pool + pubeval teacher)**
   ```bash
   python train.py \
     --algo ppo \
     --agent-type mlp \
     --model-size large \
     --device cuda \
     --teacher pubeval \
     --n-games 300000 \
     --n-epochs 5000 \
     --batch-size 8
   ```

2. **Transformer PPO variant**
   ```bash
   python train.py --algo ppo --agent-type transformer --model-size medium --device cuda
   ```

3. **Quick CPU smoke-test**
   ```bash
   python train.py --cpu-test --algo micro-a2c --model-size small
   ```

Checkpoints are written under `./checkpoints/<timestamp>/`. Pass `--resume
path/to/checkpoint.pt` to continue training, or switch `--teacher` to `gnubg`
for higher-quality DAGGER supervision (requires the `gnubg` Python package).

## ELO-based opponent sampling

During training the current agent self-plays but also spars against:

- Reference bots (`pubeval`, `random`, `gnubg`).
- The best checkpoint to date.
- Snapshots from the rolling `OpponentPool`.

Each opponent is assigned a base weight plus a live ELO rating maintained by
`EloTracker`. `compute_challenge_weight()` prefers matchups whose ELO is close
to the learner (i.e., expected score ≈ 50%), ensuring challenging yet winnable
games. Pool opponents are phased in via `compute_pool_weight_multiplier()`:
after `pool_start_games`, the trainer linearly ramps up the probability of
sampling historical snapshots until `pool_target_rate` is reached. Slow
opponents such as GNU Backgammon are marked with `slow=True`, which reduces
their batch size but keeps them in the ELO ladder. Before evaluations the
trainer can `--use-eval-lookahead` to enable a 1-ply expectiminimax search, and
`calibrate_opponent_elos()` periodically recalibrates ratings between reference
opponents to keep matchmaking fair.

## Key agents & opponents

- `ppo_agent.py`: ResMLP PPO baseline with legal-move masking, DAGGER-lite
  teacher labels, curriculum schedules, and configurable small/medium/large
  footprints.
- `ppo_conv_agent.py`: Adds a convolutional front-end that transforms the 29-d
  board into spatial feature maps before feeding a ResMLP trunk—helpful for
  capturing local structures such as primes and anchors.
- `ppo_transformer_agent.py`: Decoder-only Transformer PPO. It tokenizes game
  histories, attends over previous states, and scores legal moves with a
  dot-product head, providing stronger long-range reasoning.
- `pubeval_player.py`: Fast Numba re-implementation of Tesauro's heuristic,
  used as a teacher (`--teacher pubeval`) or as a sparring partner. Requires
  NumPy + Numba only.
- `gnu_backgammon_player.py`: Wrapper around `gnubg.best_move()`. Converts the
  environment board to TanBoard format, queries the official GNU Backgammon
  neural nets, and maps their response back into our move encoding. Install via
  `python -m pip install gnubg` to unlock `--teacher gnubg` or evaluation
  matches vs `gnubg`.

Additional utilities include `evaluate.py` (league-style evals), `opponent_pool.py`
(snapshot management), and `play_games.py` (batched rollouts). Use these scripts
to test new agents or to pit checkpoints against external opponents.
