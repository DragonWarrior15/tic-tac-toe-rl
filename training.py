from game_environment import TicTacToe
from agents import DeepQLearningAgent, NoviceAgent, RandomAgent
import numpy as np
import os
import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.display import clear_output
import pandas as pd
from utils import play_game
from tqdm import tqdm

board_size = 3
version = 'v03'
log_frequency = 500
episodes = 100000

agent = DeepQLearningAgent(board_size, use_target_net=True, buffer_size=10000)
agent_random = NoviceAgent(board_size)
env = TicTacToe()
# cold start problem, add some games to buffer for training
_, _ = play_game(agent, agent_random, env, epsilon=1, n_games=100, record=True)
# train on those games
agent.train_agent()

# main training procedure
win_counts = {'win':0, 'lose':0, 'draw':0}
loss_history = []
reward_history = []

epsilon = 0.9
decay = 0.99
epsilon_end = 0.01

# training loop
model_logs = {'iteration':[], 'reward_mean':[], 'reward_dev':[], 'wins':[],  'draws':[], 'loss':[]}
for index in tqdm(range(episodes)):
    # make small changes to the buffer and slowly train
    win_counts, current_rewards = play_game(agent, agent_random, env, epsilon=epsilon,
                        n_games=10, record=True)
    loss = agent.train_agent(batch_size=64)
    # check performance every once in a while
    if((index+1)%log_frequency == 0):
        model_logs['loss'].append(loss)
        # keep track of agent rewards_history
        current_wins, current_rewards = play_game(agent, agent_random, env, epsilon=-1, n_games=10,
                                    record=False)
        model_logs['iteration'].append(index+1)
        model_logs['reward_mean'].append(round(np.mean(current_rewards), 2))
        model_logs['reward_dev'].append(round(np.std(current_rewards), 2))
        model_logs['wins'].append(current_wins['win'])
        model_logs['draws'].append(current_wins['draw'])
        pd.DataFrame(model_logs)[['iteration', 'reward_mean', 'reward_dev', 'wins', 'draws', 'loss']].to_csv('model_logs/{:s}.csv'.format(version), index=False)

    # copy weights to target network and save models
    if((index+1)%log_frequency == 0):
        agent.update_target_net()
        agent.save_model(file_path='models/{:s}'.format(version), iteration=(index+1))
        # keep some epsilon alive for training
        epsilon = max(epsilon * decay, epsilon_end)
