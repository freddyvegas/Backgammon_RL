import backgammon
import pubeval_player as pubeval
import random_player as randomAgent
import flipped_agent as flipped_util
from utils import (
    one_hot_encoding,
    transformer_one_hot_encoding,
    flip_to_pov_plus1,
    _is_empty_move,
    _apply_move_sequence,
    append_token,
    pad_truncate_seq,
)
import numpy as np
import torch
import random

def play_games_batched(agent_obj, opponent, batch_size=8, training=True, train_config=None):
    """
    Run up to `batch_size` games in parallel and feed batched states to the network.
    Supports both PPO (with batch_score and buffer) and A2C (direct action calls).
    """
    finished = 0

    metadata = dict(train_config) if isinstance(train_config, dict) else {}
    opponent_type = metadata.get('opponent_type')
    if opponent_type is None:
        if opponent is randomAgent:
            opponent_type = 'random'
        elif opponent is pubeval:
            opponent_type = 'pubeval'
        elif opponent is agent_obj:
            opponent_type = 'self_play'
        else:
            opponent_type = 'opponent'
        metadata['opponent_type'] = opponent_type
    else:
        opponent_type = str(opponent_type)

    result_callback = metadata.get('result_callback')

    result_callback = metadata.get('result_callback')

    use_full_rewards = True
    if training and hasattr(agent_obj, 'prepare_training_context'):
        use_full_rewards = agent_obj.prepare_training_context(metadata)
    
    # Detect agent type once at the start
    is_ppo_like = hasattr(agent_obj, "batch_score") and hasattr(agent_obj, "buffer") and hasattr(agent_obj.config, "max_actions")
    cfg = getattr(agent_obj, 'config', None)
    use_raw_board_inputs = bool(getattr(cfg, 'use_raw_board_inputs', False)) if cfg else False
    raw_board_dim = getattr(cfg, 'raw_board_dim', 29) if cfg else 29
    state_dim = getattr(cfg, 'state_dim', raw_board_dim) if cfg else 293
    flag_index = getattr(cfg, 'second_roll_index', raw_board_dim) if cfg else raw_board_dim

    def encode_state_features(board_vec, second_roll_flag):
        if use_raw_board_inputs:
            feats = np.zeros(state_dim, dtype=np.float32)
            feats[:raw_board_dim] = board_vec.astype(np.float32)
            idx = min(flag_index, state_dim - 1)
            feats[idx] = 1.0 if second_roll_flag else 0.0
            return feats
        return one_hot_encoding(board_vec.astype(np.float32), second_roll_flag)

    # Keep trajectories per environment (PPO only)
    if is_ppo_like:
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
                    players[idx] = -player
                    dices[idx] = backgammon.roll_dice()
                    passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1
                continue

            if player == 1:
                agent_envs.append((idx, dice, board, pmoves, pboards))
            else:
                opponent_envs.append((idx, dice, board, pmoves, pboards))

        # ---- Process agent moves ----
        if agent_envs and is_ppo_like:
            # PPO path: batch scoring with after-states
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
                S_feat = encode_state_features(S29, nSecondRoll_now)

                # Build candidates (now state_dim wide)
                cand_feats = np.zeros(
                    (agent_obj.config.max_actions, agent_obj.config.state_dim),
                    dtype=np.float32
                )
                raw_after_states = []
                mask = np.zeros(agent_obj.config.max_actions, dtype=np.float32)

                nA = min(len(pboards), agent_obj.config.max_actions)
                pmoves_cur = pmoves[:nA]
                pboards_cur = pboards[:nA]
                for a in range(nA):
                    after29 = flip_to_pov_plus1(pboards_cur[a], 1)
                    nSecondRoll_next = (passes_left[idx] - 1) > 1
                    cand_feats[a]  = encode_state_features(after29, nSecondRoll_next)
                    raw_after_states.append(after29.astype(np.float32))
                    mask[a] = 1.0

                batch_states.append(S_feat)
                batch_cand_states.append(cand_feats)
                batch_masks.append(mask)
                after_states_np = np.stack(raw_after_states, axis=0) if raw_after_states else np.zeros((0, 29), dtype=np.float32)
                board_snapshot = board.copy()
                dice_snapshot = np.array(dice, dtype=np.int32)
                per_env_candidates.append((idx, pmoves_cur, pboards_cur, board_pov, after_states_np, board_snapshot, dice_snapshot))

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
            for row, (idx, pmoves, pboards, board_pov, raw_after_states, board_snapshot, dice_snapshot) in enumerate(per_env_candidates):
                a_idx = int(a_idxs[row])
                if a_idx >= len(pmoves):
                    a_idx = len(pmoves) - 1

                chosen_move = pmoves[a_idx]
                old_board = boards[idx].copy()
                boards[idx] = backgammon.update_board(boards[idx], chosen_move, 1)  # player=1

                # Compute reward
                reward = 0.0
                win = bool(boards[idx][27] == 15)
                loss = bool(boards[idx][28] == -15)
                if win:
                    terminal_reward = 1.0
                elif loss and use_full_rewards:
                    terminal_reward = -1.0
                else:
                    terminal_reward = 0.0

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

                    teacher_idx = -1
                    if (
                        agent_obj.has_teacher() and
                        random.random() < agent_obj.config.teacher_sample_rate
                    ):
                        teacher_idx = agent_obj.compute_teacher_index(
                            board_abs=board_snapshot,
                            dice=dice_snapshot,
                            player=1,
                            board_pov=board_pov,
                            after_states_pov=raw_after_states,
                            pmoves=pmoves,
                            mask=mask_for_this
                        )

                    # Push transition to buffer
                    per_env_rollouts[idx].append((
                        state, cand_states_for_this, mask_for_this,
                        a_idx, log_prob, value, reward,
                        1.0 if (terminal_reward != 0.0) else 0.0,
                        teacher_idx
                    ))
                    agent_obj.steps += 1

                # Check if game over
                done = backgammon.game_over(boards[idx])
                if done:
                    env_active[idx] = False
                    if result_callback:
                        outcome = 1 if win else (-1 if loss else 0)
                        if outcome:
                            result_callback(outcome)
                    if (training and not agent_obj.eval_mode and
                        hasattr(agent_obj, 'record_curriculum_result')):
                        agent_obj.record_curriculum_result(opponent_type, 1 if win else -1 if loss else 0)
                    # Flush this env's trajectory to the global PPO buffer (keep it contiguous)
                    for (S_, C_, M_, A_, LP_, V_, R_, D_, T_) in per_env_rollouts[idx]:
                        agent_obj.buffer.push(S_, C_, M_, A_, LP_, V_, R_, D_, T_)
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

        elif agent_envs:
            # A2C path: direct action calls (no batch scoring, no buffer)
            for (idx, dice, board, pmoves, pboards) in agent_envs:
                # Agent acts directly (handles its own learning inside action())
                action_cfg = metadata if metadata else None
                move = agent_obj.action(
                    board, dice, +1, 0, train=training,
                    train_config=action_cfg
                )
                if not _is_empty_move(move):
                    boards[idx] = _apply_move_sequence(board, move, +1)

                # Check terminal and pass handling
                done = backgammon.game_over(boards[idx])
                if done:
                    env_active[idx] = False
                    if result_callback:
                        if boards[idx][27] == 15:
                            result_callback(1)
                        elif boards[idx][28] == -15:
                            result_callback(-1)
                    if (training and hasattr(agent_obj, 'record_curriculum_result')):
                        outcome = 1 if boards[idx][27] == 15 else -1 if boards[idx][28] == -15 else 0
                        agent_obj.record_curriculum_result(opponent_type, outcome)
                    finished += 1
                else:
                    passes_left[idx] -= 1
                    if passes_left[idx] <= 0:
                        players[idx] = -1
                        dices[idx] = backgammon.roll_dice()
                        passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

        # ---- Process opponent moves individually ----
        for (idx, dice, board, pmoves, pboards) in opponent_envs:
            # Opponent makes a move
            if opponent is randomAgent:
                move = randomAgent.action(board, dice, -1, 0)
            elif opponent is pubeval:
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
                if is_ppo_like:
                    # PPO path: retro-credit opponent win to last agent step
                    agent_lost = bool(boards[idx][28] == -15)
                    loss_scalar = -1.0 if (agent_lost and use_full_rewards) else 0.0
                    loss_reward = loss_scalar * agent_obj.config.reward_scale
                    if training and not agent_obj.eval_mode:
                        if per_env_rollouts[idx]:
                            # Update last transition with loss reward
                            (S_, C_, M_, A_, LP_, V_, R_, D_, T_) = per_env_rollouts[idx][-1]
                            per_env_rollouts[idx][-1] = (S_, C_, M_, A_, LP_, V_, R_ + loss_reward, 1.0, T_)
                        else:
                            # Rare: agent never acted - fabricate a 1-step terminal sample
                            board_pov29 = flip_to_pov_plus1(boards[idx], 1).astype(np.float32)
                            S_feat = encode_state_features(board_pov29, False)
                            C_feats = np.zeros((agent_obj.config.max_actions, agent_obj.config.state_dim), dtype=np.float32)
                            C_feats[0] = S_feat
                            M = np.zeros(agent_obj.config.max_actions, dtype=np.float32)
                            M[0] = 1.0
                            per_env_rollouts[idx].append((S_feat, C_feats, M, 0, 0.0, 0.0, loss_reward, 1.0, -1))
                        if hasattr(agent_obj, 'record_curriculum_result') and agent_lost:
                            agent_obj.record_curriculum_result(opponent_type, -1)
                    if result_callback and agent_lost:
                        result_callback(-1)

                    # Flush this env's trajectory to buffer
                    env_active[idx] = False
                    for (S_, C_, M_, A_, LP_, V_, R_, D_, T_) in per_env_rollouts[idx]:
                        agent_obj.buffer.push(S_, C_, M_, A_, LP_, V_, R_, D_, T_)
                    per_env_rollouts[idx].clear()
                    if training and not agent_obj.eval_mode and agent_obj.buffer.is_ready():
                        agent_obj._ppo_update()
                    finished += 1
                else:
                    # A2C path: just mark as done
                    env_active[idx] = False
                    finished += 1
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

def play_games_batched_transformer(agent_obj, opponent, batch_size=8, training=True, train_config=None):
    """Batched self-play loop optimized for the transformer PPO agent."""
    finished = 0

    metadata = dict(train_config) if isinstance(train_config, dict) else {}
    opponent_type = metadata.get('opponent_type')
    if opponent_type is None:
        if opponent is randomAgent:
            opponent_type = 'random'
        elif opponent is pubeval:
            opponent_type = 'pubeval'
        elif opponent is agent_obj:
            opponent_type = 'self_play'
        else:
            opponent_type = 'opponent'
        metadata['opponent_type'] = opponent_type
    else:
        opponent_type = str(opponent_type)

    use_full_rewards = True
    if training and hasattr(agent_obj, 'prepare_training_context'):
        use_full_rewards = agent_obj.prepare_training_context(metadata)

    # Per-env containers
    env_active  = [True] * batch_size
    boards      = []
    players     = []
    dices       = []
    passes_left = []

    # Per-env trajectories (kept contiguous per env; flushed to global buffer on done)
    per_env_rollouts = [[] for _ in range(batch_size)]

    # Histories: sequence of tokens in current-player POV.
    # Pre-move tokens use actor_flag=0, post-move tokens use actor_flag=1.
    histories293 = [[] for _ in range(batch_size)]
    hist_lens    = [0  for _ in range(batch_size)]

    # Shorthands
    N    = agent_obj.config.max_seq_len
    D    = agent_obj.config.state_dim
    Amax = agent_obj.config.max_actions

    # Device for model execution
    device = agent_obj.device if hasattr(agent_obj, "device") else agent_obj.config.device

    def finalize_episode(env_idx):
        nonlocal finished
        env_active[env_idx] = False
        for (SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_, D_, T_) in per_env_rollouts[env_idx]:
            agent_obj.buffer.push(SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_, D_, T_)
        per_env_rollouts[env_idx].clear()
        if training and not agent_obj.eval_mode and agent_obj.buffer.is_ready():
            agent_obj._ppo_update()
        histories293[env_idx].clear()
        hist_lens[env_idx] = 0
        finished += 1

    # ---- Initialize games and seed history with initial token ----
    for i in range(batch_size):
        board  = backgammon.init_board()
        player = 1
        dice   = backgammon.roll_dice()

        boards.append(board)
        players.append(player)
        dices.append(dice)
        passes_left.append(2 if dice[0] == dice[1] else 1)

        board_pov = flip_to_pov_plus1(board, 1)
        append_token(
            histories293, hist_lens, i, board_pov, (passes_left[i] > 1),
            lambda b, sr, actor=0.0: transformer_one_hot_encoding(b, sr, actor),
            max_seq_len=N
        )

    # ---- Main loop ----
    while any(env_active):
        agent_envs, opponent_envs = [], []

        # Split agent/opponent turns; skip envs with no legal moves (pass handling)
        for idx in range(batch_size):
            if not env_active[idx]:
                continue
            player, dice, board = players[idx], dices[idx], boards[idx]
            pmoves, pboards = backgammon.legal_moves(board, dice, player)

            if len(pmoves) == 0:
                passes_left[idx] -= 1
                if passes_left[idx] <= 0:
                    players[idx] = -player
                    dices[idx] = backgammon.roll_dice()
                    passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1
                continue

            (agent_envs if player == 1 else opponent_envs).append((idx, dice, board, pmoves, pboards))

        # ---- Agent batch ----
        if agent_envs:
            batch_states = []
            batch_cand_states = []
            batch_masks = []
            per_env_candidates = []
            seq_batch = []
            seq_len_batch = []

            for (idx, dice, board, pmoves, pboards) in agent_envs:
                board_pov = flip_to_pov_plus1(board, 1).astype(np.float32)
                moves_left = passes_left[idx]

                append_token(
                    histories293, hist_lens, idx, board_pov, (moves_left > 1),
                    lambda b, sr, actor=0.0: transformer_one_hot_encoding(b, sr, actor),
                    max_seq_len=N
                )

                cand_feats = np.zeros((Amax, D), dtype=np.float32)
                mask = np.zeros(Amax, dtype=np.float32)
                raw_after_states = []

                nA = min(len(pboards), Amax)
                pmoves_cur = pmoves[:nA]
                pboards_cur = pboards[:nA]
                nSecondRoll_next = ((passes_left[idx] - 1) > 1)
                for a in range(nA):
                    after29 = flip_to_pov_plus1(pboards_cur[a], 1).astype(np.float32)
                    cand_feats[a] = transformer_one_hot_encoding(after29, nSecondRoll_next, actor_flag=1.0)
                    raw_after_states.append(after29)
                    mask[a] = 1.0

                batch_states.append(transformer_one_hot_encoding(board_pov, (moves_left > 1), actor_flag=0.0))
                batch_cand_states.append(cand_feats)
                batch_masks.append(mask)
                after_np = np.stack(raw_after_states, axis=0) if raw_after_states else np.zeros((0, 29), dtype=np.float32)
                board_snapshot = board.copy()
                dice_snapshot = np.array(dice, dtype=np.int32)
                per_env_candidates.append((idx, pmoves_cur, pboards_cur, board_pov, after_np, board_snapshot, dice_snapshot))

                seq_pad, seq_len = pad_truncate_seq(
                    histories293[idx], N, D
                )
                seq_batch.append(seq_pad)
                seq_len_batch.append(seq_len)

            states_np = np.stack(batch_states, axis=0)
            cand_states_np = np.stack(batch_cand_states, axis=0)
            masks_np = np.stack(batch_masks, axis=0)
            seq_batch_np = np.stack(seq_batch, axis=0)
            seq_len_np = np.asarray(seq_len_batch, dtype=np.int64)

            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            cand_states_t = torch.as_tensor(cand_states_np, dtype=torch.float32, device=device)
            masks_t = torch.as_tensor(masks_np, dtype=torch.float32, device=device)
            hist_pad_t = torch.as_tensor(seq_batch_np, dtype=torch.float32, device=device)
            hist_len_t = torch.as_tensor(seq_len_np, dtype=torch.int64, device=device)

            with torch.no_grad():
                logits, values = agent_obj.batch_score(
                    states_t, cand_states_t, masks_t,
                    histories293=hist_pad_t,
                    histories_len=hist_len_t
                )

            # Action selection
            if training and not agent_obj.eval_mode:
                probs = torch.softmax(logits, dim=-1)
                a_idxs = torch.multinomial(probs, num_samples=1).squeeze(1)
                a_idxs_long = a_idxs.long()
                log_probs_t = torch.log(
                    torch.gather(probs, 1, a_idxs_long.unsqueeze(1)).squeeze(1) + 1e-9
                )
                a_idxs    = a_idxs.tolist()
                log_probs = log_probs_t.tolist()
            else:
                a_idxs    = torch.argmax(logits, dim=-1).tolist()
                log_probs = [0.0] * len(a_idxs)

            # Apply actions and record transitions
            for row, (idx, pmoves, pboards, board_pov, raw_after_states, board_snapshot, dice_snapshot) in enumerate(per_env_candidates):
                a_idx = int(a_idxs[row])
                if a_idx >= len(pmoves):  # clamp against padding
                    a_idx = len(pmoves) - 1

                chosen_move = pmoves[a_idx]
                old_board = boards[idx].copy()
                boards[idx] = backgammon.update_board(boards[idx], chosen_move, 1)

                # Rewards
                win = bool(boards[idx][27] == 15)
                loss = bool(boards[idx][28] == -15)
                if win:
                    terminal_reward = 1.0
                elif loss and use_full_rewards:
                    terminal_reward = -1.0
                else:
                    terminal_reward = 0.0
                shaped_reward = 0.0
                if training and not agent_obj.eval_mode and agent_obj.config.use_reward_shaping:
                    board_pov_old = flip_to_pov_plus1(old_board, 1)
                    board_pov_new = flip_to_pov_plus1(boards[idx], 1)
                    shaped_reward = agent_obj._compute_shaped_reward(board_pov_old, board_pov_new)
                reward = (terminal_reward + shaped_reward) * agent_obj.config.reward_scale
                done = 1.0 if backgammon.game_over(boards[idx]) else 0.0
                if done and training and not agent_obj.eval_mode and hasattr(agent_obj, 'record_curriculum_result'):
                    agent_obj.record_curriculum_result(opponent_type, 1 if win else -1 if loss else 0)
                if done and result_callback:
                    if win:
                        result_callback(1)
                    elif loss:
                        result_callback(-1)

                # Append the post-action observation token
                append_token(
                    histories293, hist_lens, idx,
                    flip_to_pov_plus1(boards[idx], 1), False,
                    lambda b, sr, actor=1.0: transformer_one_hot_encoding(b, sr, actor),
                    max_seq_len=N
                )

                # Push transformer tuple to per-env rollout (training only)
                if training and not agent_obj.eval_mode:
                    seq_padded_np, seq_len = pad_truncate_seq(
                        histories293[idx], N, D
                    )
                    value = values[row].item() if hasattr(values[row], 'item') else float(values[row])
                    cand_np = batch_cand_states[row]
                    mask_np = batch_masks[row]
                    teacher_idx = -1
                    if (
                        agent_obj.has_teacher() and
                        random.random() < agent_obj.config.teacher_sample_rate
                    ):
                        teacher_idx = agent_obj.compute_teacher_index(
                            board_abs=board_snapshot,
                            dice=dice_snapshot,
                            player=1,
                            board_pov=board_pov,
                            after_states_pov=raw_after_states,
                            pmoves=pmoves,
                            mask=mask_np
                        )
                    per_env_rollouts[idx].append((
                        seq_padded_np, int(seq_len),
                        cand_np, mask_np,
                        a_idx, float(log_probs[row]), float(value),
                        float(reward), float(done), teacher_idx
                    ))
                    agent_obj.steps += 1

                # Episode end?
                if backgammon.game_over(boards[idx]):
                    finalize_episode(idx)
                else:
                    passes_left[idx] -= 1
                    if passes_left[idx] <= 0:
                        players[idx] = -1
                        dices[idx] = backgammon.roll_dice()
                        passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

        # ---- Opponent (step individually) ----
        if opponent_envs:
            if opponent is agent_obj:
                # Mirror self-play using the same policy (greedy) without buffer writes
                sp_states = []
                sp_cands = []
                sp_masks = []
                env_refs = []
                for (idx, dice, board, pmoves, pboards) in opponent_envs:
                    board_pov = flip_to_pov_plus1(board, -1).astype(np.float32)
                    moves_left = passes_left[idx]

                    append_token(
                        histories293, hist_lens, idx,
                        flip_to_pov_plus1(board, -1), (moves_left > 1),
                        lambda b, sr, actor=0.0: transformer_one_hot_encoding(b, sr, actor),
                        max_seq_len=N
                    )

                    cand_feats = np.zeros((Amax, D), dtype=np.float32)
                    mask = np.zeros(Amax, dtype=np.float32)
                    nA = min(len(pboards), Amax)
                    nSecondRoll_next = ((passes_left[idx] - 1) > 1)
                    for a in range(nA):
                        after29 = flip_to_pov_plus1(pboards[a], -1).astype(np.float32)
                        cand_feats[a] = transformer_one_hot_encoding(after29, nSecondRoll_next, actor_flag=1.0)
                        mask[a] = 1.0

                    sp_states.append(transformer_one_hot_encoding(board_pov, (moves_left > 1), actor_flag=0.0))
                    sp_cands.append(cand_feats)
                    sp_masks.append(mask)
                    env_refs.append((idx, pmoves, pboards))

                states_t = torch.as_tensor(np.stack(sp_states, axis=0), dtype=torch.float32, device=device)
                cand_t = torch.as_tensor(np.stack(sp_cands, axis=0), dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(np.stack(sp_masks, axis=0), dtype=torch.float32, device=device)
                # Histories already padded for agent usage
                hist_batch = []
                hist_len = []
                for idx, _, _ in env_refs:
                    seq_np, seq_len = pad_truncate_seq(
                        histories293[idx], N, D
                    )
                    hist_batch.append(seq_np)
                    hist_len.append(seq_len)
                hist_t = torch.as_tensor(np.stack(hist_batch, axis=0), dtype=torch.float32, device=device)
                len_t = torch.as_tensor(hist_len, dtype=torch.int64, device=device)

                with torch.no_grad():
                    logits, _ = agent_obj.batch_score(states_t, cand_t, mask_t, histories293=hist_t, histories_len=len_t)
                a_idxs = torch.argmax(logits, dim=-1).tolist()

                for row, (idx, pmoves, pboards) in enumerate(env_refs):
                    a_idx = min(a_idxs[row], len(pmoves) - 1)
                    move = pmoves[a_idx]
                    if not _is_empty_move(move):
                        boards[idx] = _apply_move_sequence(boards[idx], move, -1)

                    append_token(
                        histories293, hist_lens, idx,
                        flip_to_pov_plus1(boards[idx], -1), False,
                        lambda b, sr, actor=1.0: transformer_one_hot_encoding(b, sr, actor),
                        max_seq_len=N
                    )

                    if backgammon.game_over(boards[idx]):
                        if training and not agent_obj.eval_mode:
                            loss_scalar = -1.0 if (boards[idx][28] == -15 and use_full_rewards) else 0.0
                            loss_reward = loss_scalar * agent_obj.config.reward_scale
                            if per_env_rollouts[idx]:
                                (SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_, D_, T_) = per_env_rollouts[idx][-1]
                                per_env_rollouts[idx][-1] = (SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_ + loss_reward, 1.0, T_)
                            else:
                                seq_padded_np, seq_len = pad_truncate_seq(
                                    histories293[idx], N, D
                                )
                                C_feats = np.zeros((Amax, D), dtype=np.float32)
                                C_feats[0] = seq_padded_np[max(0, seq_len - 1)]
                                M = np.zeros(Amax, dtype=np.float32); M[0] = 1.0
                                per_env_rollouts[idx].append(
                                    (seq_padded_np, int(seq_len), C_feats, M, 0, 0.0, 0.0, float(loss_reward), 1.0, -1)
                                )
                            if hasattr(agent_obj, 'record_curriculum_result') and boards[idx][28] == -15:
                                agent_obj.record_curriculum_result(opponent_type, -1)
                        if result_callback and boards[idx][28] == -15:
                            result_callback(-1)
                        finalize_episode(idx)
                    else:
                        passes_left[idx] -= 1
                        if passes_left[idx] <= 0:
                            players[idx] = 1
                            dices[idx] = backgammon.roll_dice()
                            passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

            else:
                for (idx, dice, board, pmoves, pboards) in opponent_envs:
                    if opponent == randomAgent:
                        move = randomAgent.action(board, dice, -1, 0)
                    elif opponent == pubeval:
                        move = pubeval.action(board, dice, -1, 0)
                    elif hasattr(opponent, 'action'):
                        move = opponent.action(board, dice, -1, 0, train=False)
                    else:
                        import random as py_random
                        move = pmoves[py_random.randint(0, len(pmoves) - 1)]

                    if not _is_empty_move(move):
                        boards[idx] = _apply_move_sequence(board, move, -1)

                    append_token(
                        histories293, hist_lens, idx,
                        flip_to_pov_plus1(boards[idx], -1), False,
                        lambda b, sr, actor=1.0: transformer_one_hot_encoding(b, sr, actor),
                        max_seq_len=N
                    )

                    if backgammon.game_over(boards[idx]):
                        if training and not agent_obj.eval_mode:
                            loss_scalar = -1.0 if (boards[idx][28] == -15 and use_full_rewards) else 0.0
                            loss_reward = loss_scalar * agent_obj.config.reward_scale
                            if per_env_rollouts[idx]:
                                (SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_, D_, T_) = per_env_rollouts[idx][-1]
                                per_env_rollouts[idx][-1] = (SEQ_, SEQL_, C_, M_, A_, LP_, V_, R_ + loss_reward, 1.0, T_)
                            else:
                                seq_padded_np, seq_len = pad_truncate_seq(
                                    histories293[idx], N, D
                                )
                                C_feats = np.zeros((Amax, D), dtype=np.float32)
                                C_feats[0] = seq_padded_np[max(0, seq_len - 1)]
                                M = np.zeros(Amax, dtype=np.float32); M[0] = 1.0
                                per_env_rollouts[idx].append(
                                    (seq_padded_np, int(seq_len), C_feats, M, 0, 0.0, 0.0, float(loss_reward), 1.0, -1)
                                )
                            if hasattr(agent_obj, 'record_curriculum_result') and boards[idx][28] == -15:
                                agent_obj.record_curriculum_result(opponent_type, -1)
                        if result_callback and boards[idx][28] == -15:
                            result_callback(-1)
                        finalize_episode(idx)
                    else:
                        passes_left[idx] -= 1
                        if passes_left[idx] <= 0:
                            players[idx] = 1
                            dices[idx] = backgammon.roll_dice()
                            passes_left[idx] = 2 if dices[idx][0] == dices[idx][1] else 1

    return finished

def play_one_game(agent1, agent2, training=False, commentary=False,
                  use_lookahead=False, lookahead_k=3):
    """Play one game with optional top-k lookahead (now symmetric)."""
    board = backgammon.init_board()
    player = np.random.randint(2) * 2 - 1
    lookahead_cfg = None
    if use_lookahead and not training:
        lookahead_cfg = {
            'use_lookahead': True,
            'lookahead_k': lookahead_k
        }

    if hasattr(agent1, "episode_start"): agent1.episode_start()
    if hasattr(agent2, "episode_start"): agent2.episode_start()

    while not backgammon.game_over(board) and not backgammon.check_for_error(board):
        dice = backgammon.roll_dice()
        if commentary:
            print(f"player {player}, dice {dice}")

        for r in range(1 + int(dice[0] == dice[1])):
            board_copy = board.copy()

            if player == 1:
                cfg = lookahead_cfg if (lookahead_cfg and hasattr(agent1, '_evaluate_moves_lookahead')) else None
                move = agent1.action(board_copy, dice, player, i=r, train=training, train_config=cfg)
            else:
                cfg = lookahead_cfg if (lookahead_cfg and hasattr(agent2, '_evaluate_moves_lookahead')) else None
                move = agent2.action(board_copy, dice, player, i=r, train=training, train_config=cfg)

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
