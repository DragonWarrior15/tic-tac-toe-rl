import numpy as np

# class for the tic tac toe environment
class TicTacToe:
    # initialization
    def __init__(self, board_size = 3):
        assert isinstance(board_size, int), "board size should be integer, got {} instead".format(isinstance(board_size))
        self._size = board_size
        self._board = np.zeros(self._size ** 2, dtype = np.int8)

        self._rewards = {
            'win'     : 10,
            'draw'    : 1,
            'lose'    : -1,
            'invalid' : -10
        }

        self.initialize_vars()

    def get_max_reward(self):
        return(max([x for x in self._rewards.values()]))

    def initialize_vars(self):
        # indicator to prevent any further moves
        self._is_done = 0
        self._start_move = 0
        self._next_move = 0

        # store the row, column and diag sums handy for quick winner check
        # for checking if board is complete
        self._zero_counts = len(self._board)

        # row sums
        self._row_sums = [0] * self._size

        # column sums
        self._col_sums = [0] * self._size

        self._diag_sums = [0, 0] # for \ and /

    # print routine
    def print_board(self):
        str_print = ''
        for i in range(self._size):
            str_print += ' | '.join(list(map(str, self._board[i*self._size : (i+1)*self._size]))) + '\n'

        print(str_print)

    def reset(self):
        self._board = np.zeros_like(self._board)
        self.initialize_vars()
        return self._board.copy()

    # provide whether to put X or O
    # position is a single integer in range
    def step(self, num, position):
        reward = {-1:0, 1:0}
        if(self.is_valid_move(num, position)):
            self._board[position] = num
        else:
            reward[num] = self._rewards['invalid']

            # terminate game to suppress invalid moves
            return (self._board, reward, 0 ,1)

        # update column, row and diagonal sums
        self._zero_counts -= 1
        self._row_sums[position//self._size] += num
        self._col_sums[position%self._size] += num
        self._diag_sums[0] += num if(position % (self._size+1) == 0) else 0
        self._diag_sums[1] += num if(position % (self._size-1) == 0 \
                                     and position != 0) else 0

        reward, winner, done = self.is_end()

        # reward is number, done is 0 or 1
        return (self._board.copy(), reward, winner, done)

    # check if valid move has been made
    def is_valid_move(self, num, position):
        assert isinstance(position, int), \
          "position should be of type int, but is of type {} instead".format(type(position))
        assert 0 <= position and position < len(self._board), \
          "position should be in range({}, {}), but is {} instead".format(0, len(self._board), position)
        assert num in [-1, 1], "played number should be -1 or 1, but is {} instead".format(num)

        if(self._is_done == 1):
            # game has ended
            print("game has ended already, reset environment to proceed")
            return False

        if(self._zero_counts < len(self._board)):
            if(num != self._next_move):
                print("illegal turn")
                return False

        if(self._zero_counts == len(self._board)):
            # is the first move
            self._start_move = num
            self._next_move = -1 * num

        if(self._board[position] != 0):
            return False

        self._next_move = -1 * num

        return True

    # check if the game has ended
    def is_end(self):
        # 2 player reward
        reward = {-1:0, 1:0}

        # check for winner
        for i in range(self._size):
            # check in rows
            if(abs(self._row_sums[i]) == self._size):
                reward, winner = self.set_winner_reward(self._row_sums[i])
                return reward, winner, 1

            # check in columns
            if(abs(self._col_sums[i]) == self._size):
                reward, winner = self.set_winner_reward(self._col_sums[i])
                return reward, winner, 1

        # check in diagonals
        for i in [0, 1]:
            if(abs(self._diag_sums[i]) == self._size):
                reward, winner = self.set_winner_reward(self._diag_sums[i])
                return reward, winner, 1

        # check if board is complete, then sure tie
        # as winner part has been completed without finding one
        if(self._zero_counts == 0):
            is_tie = True
            reward[-1] = self._rewards['draw']
            reward[1] = self._rewards['draw']

            return reward, 0, 1

        return reward, 0, 0

    def set_winner_reward(self, some_sum):
        assert abs(some_sum) == self._size, "some bug in code, check please"

        reward = {-1:0, 1:0}
        winner = -1 if some_sum < 0 else 1
        reward[winner] = self._rewards['win']
        reward[-1 * winner] = self._rewards['lose']

        return reward, winner
