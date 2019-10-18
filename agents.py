import numpy as np

# replay buffer
from replayBuffer import ReplayBuffer

# import tensor libraries
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

# random agent to help with training
class RandomAgent:
    def __init__(self, board_size):
        self._board_size = board_size

    def move(self, move_type, board):
        assert (board == 0).sum() > 0, "Invalid board for random agent"
        return np.argmax(np.random.random(board.shape) * (board == 0))

# novice player is slightly smart, will try to form a 3 in a row
# or block, as we want the agent to learn these strategies
# strategy is not really scalable to boards of sizes more than 3
class NoviceAgent:
    def __init__(self, board_size):
        self._board_size = board_size

    def move(self, move_type, board):
        board_mod = board.reshape(self._board_size, self._board_size)
        # check if either move_type or opposite are being formed
        for i in range(self._board_size):
            # check in row
            if(abs(board_mod[i, :].sum()) == (self._board_size-1)):
                for j in range(self._board_size):
                    if(board_mod[i, j] == 0):
                        return i*self._board_size + j

            # check in column
            if(abs(board_mod[:, i].sum()) == (self._board_size-1)):
                for j in range(self._board_size):
                    if(board_mod[j, i] == 0):
                        return j*self._board_size + i

        # check the \ diagonal
        if(abs(sum([board_mod[i, i] for i in range(self._board_size)])) == (self._board_size-1)):
            for i in range(self._board_size):
                if(board_mod[i, i] == 0):
                    return i*self._board_size + i

        # check the / diagonal
        if(abs(sum([board_mod[i, self._board_size-1 - i] for i in range(self._board_size)])) == (self._board_size-1)):
            for i in range(self._board_size):
                if(board_mod[i, self._board_size - 1 - i] == 0):
                    return i*self._board_size + self._board_size - 1 - i

        # if none of the above combo is possible, return a random cell
        valid_moves = [i for i in range(len(board)) if board[i] == 0]
        return int(np.random.choice(valid_moves, 1)[0])

# class for the rl agent playing tic tac toe
# aim is to force the bot to learn rules as well
class DeepQLearningAgent:
    # initialization function
    def __init__(self, board_size, gamma=0.9, buffer_size=3000, use_target_net=False):
        assert 0 <= gamma and gamma <= 1, "gamma should be in 0 to 1, got {}".format(gamma)

        self._board_size = board_size
        self._gamma = gamma

        self._buffer = ReplayBuffer(buffer_size)
        self._buffer_size = buffer_size

        self._input_shape = (self._board_size, self._board_size, 1)
        self._model = self.agent_model()
        self._use_target_net = use_target_net
        if(use_target_net):
            self._target_net = self.agent_model()
            self.update_target_net()

    # get action value
    def get_qvalues(self, board, model = None):
        # board is assumed to be of shape 1, 1 * board_size**2
        if model is None:
            model = self._model
        q_values = model.predict(board)
        return q_values

    # get the action using greedy policy
    def move(self, move_type, board):
        q_values = self.get_qvalues(self.transform_board(move_type, board), self._model)
        action = int(np.argmax(q_values))
        return action

    # transform the input board as relevant for the model
    def transform_board(self, move_type, board):
        if(False):
            # this will flatten the input
            return_board = np.zeros(board.shape[0]+1)
            return_board[-1] = move_type
            return_board[:-1] = board
            return_board.reshape(1, -1)
        else:
            # this will maintain a 2d shape
            # add a function to transform the board into matrix of 2 x size x size
            # first plane corresponds to current player, and second to opposite player
            return_board = np.zeros((1, self._board_size, self._board_size, 2))
            return_board[0, :,:,0] = (board.reshape((self._board_size, self._board_size)) == move_type).astype(int)
            return_board[0, :,:,1] = (board.reshape((self._board_size, self._board_size)) == (move_type * -1)).astype(int)
        return return_board

    def agent_model(self):
        if(False):
            # this is the flattened version of the board as input
            # move type added to the board itself in the input
            input_board = Input((1 + self._board_size ** 2,))
            # total rows + columns + diagonals is total units
            x = Dense(2 * (self._board_size ** 2), activation = 'relu')(input_board)
            x = Dense(self._board_size ** 2, activation = 'relu')(x)
            x = Dense(self._board_size ** 2, activation = 'relu')(x)
            out = Dense(self._board_size ** 2, activation = 'linear', name = 'action_values')(x)
        else:
            # this preserves the 2d shape of the board
            input_board = Input((self._board_size, self._board_size, 2, ))
            x = Conv2D(18, (3,3), activation = 'relu', data_format='channels_last')(input_board)
            x = Flatten()(x)
            x = Dense(9, activation = 'relu')(x)
            out = Dense(self._board_size**2, activation = 'linear', name = 'action_values')(x)

        model = Model(inputs = input_board, outputs = out)
        model.compile(optimizer = RMSprop(5e-5), loss = 'mean_squared_error')

        return model

    # save the current models, note that all models are saved
    def save_model(self, file_path = '', iteration = None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.save("{}/model_{:04d}_target.h5".format(file_path, iteration))

    # load any existing models
    def load_model(self, file_path = '', iteration = None):
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        try:
            self._model = load_model("{}/model_{:04d}.h5".format(file_path, iteration))
            if(self._use_target_net):
                self._target_net  = load_model("{}/model_{:04d}_target.h5".format(file_path, iteration))
        except FileNotFoundError:
            print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_model(self):
        print('Model')
        print(self._model.summary())
        print('Target Network')
        print(self._target_net.summary())

    # add current game step to the replay buffer
    # no processing happens here, discounted rewards should be calculated
    # when training as target network latest at that point needs to be used
    def add_to_buffer(self, move_type, board, next_board, reward, action, done):
        # one hot encoding to convert the discounted rewards
        one_hot_action = np.zeros((1, self._board_size ** 2))
        one_hot_action[0, action] = 1
        '''
        # use if oversampling required
        add_times = 1
        if(done and reward > 0):
            add_times = 10
        for _ in range(add_times):
            self._buffer.add_data([board, move_type,
                            one_hot_action, discounted_reward])
        '''
        # append move type to board
        board_mod = self.transform_board(move_type, board)
        # append move type to next board
        next_board_mod = self.transform_board(move_type, next_board)

        self._buffer.add_to_buffer([board_mod, one_hot_action,
                            reward, next_board_mod, done])

    def reset_buffer(self, buffer_size = None):
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBuffer(self._buffer_size)

    def train_agent(self, batch_size=64, verbose = 0):
        # calculate the discounted rewards on the fly
        s, a, r, next_s, done = self._buffer.sample(batch_size)
        not_done = 1 - done
        current_model = self._target_net if self._use_target_net else self._model
        discounted_reward = r + (self._gamma * np.max(self.get_qvalues(next_s, current_model), axis = 1).reshape(-1, 1)) * not_done
        # calculate target using one hot action
        target = self.get_qvalues(s, self._model)
        target = discounted_reward * a + target * (1 - a)
        # train the model
        loss = self._model.train_on_batch(s, target)
        loss = round(loss, 5)
        return loss
    # target network outputs is what we try to predict
    # this network is static for a while and serves as "ground truth"
    def update_target_net(self):
        if(self._use_target_net):
            self._target_net.set_weights(self._model.get_weights())

    # to update weights between competing agents
    def copy_weights_from_agent(self, agent_for_copy):
        assert isinstance(agent_for_copy, Agent), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model.get_weights())    
