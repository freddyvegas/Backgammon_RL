# test_match.py
import numpy as np
import time

import backgammon                      
import random_player as randomAgent    
import pubeval_player as pubeval       

def _is_empty_move(move):
    # Agents return [] (list) when they must pass. Numpy arrays aren’t “== []”.
    if move is None:
        return True
    if isinstance(move, (list, tuple)):
        return len(move) == 0
    if isinstance(move, np.ndarray):
        return move.size == 0
    return False

def play_one_game(agent1, agent2, commentary=False):
    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1  # +1 or -1

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()
        if commentary:
            print(f"player {player}, dice {dice}")

        # one turn: if doubles, play twice
        for _ in range(1 + int(dice[0] == dice[1])):
            board_copy = board.copy()

            move = agent1.action(board_copy, dice, player, i=0) if player == 1 \
                else agent2.action(board_copy, dice, player, i=0)

            if _is_empty_move(move):
                # no legal move (pass)
                continue

            # Normalize to numpy array
            mv = np.asarray(move, dtype=np.int32)

            # Our Backgammon.update_board supports either a single [2,] move
            # or a sequence shape (k,2). Call it ONCE per turn.
            board = backgammon.update_board(board, mv, player)

        player = -player

    return -1 * player

def main(n_games=2000):
    start = time.time()
    wins = {1: 0, -1: 0}
    for g in range(n_games):
        w = play_one_game(randomAgent, pubeval, commentary=False)
        wins[w] += 1
        if (g+1) % 200 == 0:
            print(f"[{g+1}/{n_games}] interim win-rate (Random as P1 vs Pubeval): "
                  f"{wins[1]/(g+1):.3f}")

    dur = time.time() - start
    print(f"\nResults over {n_games} games:")
    print(f"  Random wins:  {wins[1]}")
    print(f"  Pubeval wins: {wins[-1]}")
    print(f"Wall time: {dur:.2f}s  |  {dur/n_games:.6f}s/game")

if __name__ == "__main__":
    main()

