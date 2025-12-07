from tqdm import tqdm
from pathlib import Path
from play_games import play_one_game

def evaluate(agent_mod, evaluation_agent, n_eval, label="", debug_sides=False,
             use_lookahead=False, lookahead_k=3, quiet=False):
    """Evaluate agent with fixed side alternation."""
    wins = 0
    wins_as_p1 = 0
    wins_as_p2 = 0
    games_as_p1 = 0
    games_as_p2 = 0

    for g in tqdm(range(n_eval)):
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

    if not quiet:
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
