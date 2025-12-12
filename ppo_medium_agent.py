"""
Friðrik's Medium PPO Backgammon Agent
Compatible with tournament.py interface
Trained: November 11, 2024 (14.1M steps)
"""
import sys
import importlib.util

# Load November 11th architecture
spec = importlib.util.spec_from_file_location("ppo_agent_old", "ppo_agent_nov11.py")
ppo_agent_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo_agent_old)

# Global agent instance (loaded once)
_agent = None

def initialize(checkpoint_path="best_so_far_medium.pt", device="cpu"):
    """Call this once at tournament start"""
    global _agent
    if _agent is None:
        _agent = ppo_agent_old.PPOAgent(
            config=ppo_agent_old.get_config("medium"),
            device=device
        )
        _agent.load(checkpoint_path, map_location=device, load_optimizer=False)
        _agent.set_eval_mode(True)
    return _agent

def action(board, dice, player, roll_index=0, use_lookahead=False):
    """
    Tournament-compatible interface
    
    Args:
        board: 29-entry NumPy array (absolute coordinates)
        dice: tuple (d1, d2)
        player: +1 or -1
        roll_index: dice consumed this turn
        use_lookahead: enable 1-ply lookahead (slower but stronger)
    
    Returns:
        list of (from, to) tuples, or [] if no legal move
    """
    if _agent is None:
        initialize()
    
    train_config = {"use_lookahead": use_lookahead, "lookahead_k": 1} if use_lookahead else None
    
    return _agent.action(
        board.copy(),
        dice,
        player,
        roll_index,
        train=False,
        train_config=train_config
    )

# Auto-initialize on import
initialize()
