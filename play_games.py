import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
from utils import one_hot_encoding, flip_to_pov_plus1, _is_empty_move, _apply_move_sequence
import numpy as np
import torch

def play_games_batched(agent_obj, opponent, batch_size=8, training=True):
    """
    Run up to `batch_size` games in parallel and feed batched states to the network.
    Uses the same reward shaping and rollout buffer logic the agent expects.
    """
    finished = 0
    # Keep trajectories per environment so GAE never mixes them
    per_env_rollouts = [[] for _ in range(batch_size)]

    # Each slot holds the current env state or None if finished
    env_active = [True] * batch_size
    boards     = []
    players    = []
    dices      = []
    passes_left = [] 

    # Initialize games
    for _ in range(batch_size):
        board = backgammon.init_board()
        player = 1  # we'll always feed +1 POV into the net
        dice = backgammon.roll_dice()
        boards.append(board)
        players.append(player)
        dices.append(dice)
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

            # Action selection
            if training and not agent_obj.eval_mode:
                probs = torch.softmax(logits, dim=-1)
                a_idxs = torch.multinomial(probs, num_samples=1).squeeze(1)
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

                    # Push transition to buffer
                    per_env_rollouts[idx].append((
                        state, cand_states_for_this, mask_for_this,
                        a_idx, log_prob, value, reward,
                        1.0 if (terminal_reward != 0.0) else 0.0
                    ))
                    agent_obj.steps += 1

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
                else:
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
                # Decrement passes
                passes_left[idx] -= 1
                if passes_left[idx] <= 0:
                    # Switch back to agent and roll new dice
                    players[idx] = 1
                    dices[idx] = backgammon.roll_dice()
                    passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

    return finished


# --- Top-k one-ply lookahead (POV-aware) ---
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
