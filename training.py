import importlib
import game_environment
game_environment = importlib.reload(game_environment)

import agents
agents = importlib.reload(agents)

import numpy as np
import keras.backend as K
import os
import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.display import clear_output
import pandas as pd

# a function to play a few games and add data to buffer
# opponent for the agent is assumed to be part of the env itself
# this means that, for any current state, next state will be the board after
# the opponent has also played its move
# this is a common api to both simulate games and record if needed
def play_and_record(agent_main, agent_env, env, n_games = 100, record = True):

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
            a = agent_env.move(turn, s) if turn == env_move else agent_main.move(turn, s)
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

    axs[0].plot(pd.ewma(np.array(rewards), span=50, min_periods=50))
    axs[0].set_title('Agent {}, EWMA'.format('rewards'))

    axs[1].plot(pd.ewma(np.array(losses), span=50, min_periods=50))
    axs[1].set_title('Agent {}, EWMA'.format('losses'))

    plt.show()

K.clear_session()
agent = agents.Agent(3, epsilon = 0.5, use_target_net = True)
agent_random = agents.NoviceAgent(3)
env = game_environment.TicTacToe()
# cold start problem, add some games to buffer for trainin
_, _ = play_and_record(agent, agent_random, env, n_games = 10000)
# train on those games
agent.train_agent()

# main training procedure
win_counts = {'win':0, 'lose':0, 'draw':0}
loss_history = []
reward_history = []

epsilon = 0.9

# simulate games and observe progress
for i in range(100):
    # play and record
    epsilon *= 0.99

    epsilon = max(epsilon, 0.1)

    # train the agents and evaluate without epsilon
    agent.set_epsilon(0)
    win_count, rewards = play_and_record(agent, agent_random, env, 1, record=False)
    for k in win_counts:
        win_counts[k] += win_count[k]
    reward_history.append(rewards[-1])

    # print(env.print_board())

    agent.set_epsilon(epsilon)

    _, _ = play_and_record(agent, agent_random, env, 1000)
    # train agents
    loss = agent.train_agent(epochs = 10)
    loss_history.append(loss)

    # if(i%10 == 0):
    # clear_output(True)
    plot_rewards_losses(reward_history, loss_history)

    # update target weights every 50 evaluations
    if(i % 20 == 0):
        agent.update_target_net()

    # save the model every 50 iterations
    if(i%50 == 0):
        agent.save_model(file_path = 'models/', iteration = i)
