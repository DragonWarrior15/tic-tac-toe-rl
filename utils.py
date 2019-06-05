import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a function to play a few games and add data to buffer
# opponent for the agent is assumed to be part of the env itself
# this means that, for any current state, next state will be the board after
# the opponent has also played its move
# this is a common api to both simulate games and record if needed
def play_game(agent_main, agent_env, env, epsilon=0.1, n_games=100, record=True):

    win_counts = {'win':0, 'lose':0, 'draw':0}
    reward_history = []

    for _ in range(n_games):
        env_move = -1
        # randomly assign starting moves
        if(np.random.random() <= 0.5):
            env_move = 1

        s = env.reset()
        done = 0
        turn = -1

        reward_history.append(0)

        # move_type, board, next_board, reward, action, done
        # info given to buffer to add
        buffer = [0] * 6

        # play until end
        while(not done):
            if(turn != env_move):
                if(np.random.random() < epsilon):
                    a = np.random.randint(0, env.get_board_size())
                else:
                    a = agent_env.move(turn, s)
            else:
                a = agent_env.move(turn, s)
            next_s, r, winner, done = env.step(turn, a)
            reward_history[-1] += r[-1 * env_move]

            if(record):
                # if this was agent's move, initialize the new buffer entry
                if(turn != env_move):
                    buffer = [0] * 6
                    # this is the input state for the main agent
                    buffer[0] = turn
                    buffer[1] = s.copy()
                    buffer[3] += r[turn]
                    buffer[4] = a
                    buffer[5] = 0

                    # if this was agent's move and game is over
                    if(done):
                        buffer[2] = next_s.copy()
                        buffer[5] = done
                        agent_main.add_to_buffer(*buffer)

                # if this was env's move, and the buffer has already been initialized
                if(turn == env_move and buffer[0] != 0):
                    buffer[2] = next_s.copy()
                    buffer[3] += r[turn * -1]
                    buffer[5] = done
                    agent_main.add_to_buffer(*buffer)

            s = next_s.copy()
            turn *= -1

        if(winner == 0):
            win_counts['draw'] += 1
        elif(winner == env_move):
            win_counts['lose'] += 1
        else:
            win_counts['win'] += 1

    return win_counts, reward_history

# plotting reward and losses
def plot_rewards_losses(rewards, losses):
    fig, axs = plt.subplots(1, 2, figsize=(17, 8))

    axs[0].plot(np.array(rewards))
    axs[0].set_title('Agent {}'.format('rewards'))

    axs[1].plot(np.array(losses))
    axs[1].set_title('Agent {}'.format('losses'))

    plt.show()
