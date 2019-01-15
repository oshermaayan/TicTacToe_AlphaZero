from typing import Dict

import numpy as np
import tkinter as tk
import copy
##from FeatureExtractor import *
from myLearningAgents import *
import pickle as pickle    # cPickle is for Python 2.x only; in Python 3, simply "import pickle" and the accelerated version will be used automatically if available
import math
from numpy.core.multiarray import ndarray
import csv


class Game:
    def __init__(self, master, player1, player2, Q_learn=None, Q={}, alpha=10e-6, gamma=0.9, board_size=3, streak_size=3):
        frame = tk.Frame()
        frame.grid()
        self.master = master
        master.title("Tic Tac Toe")

        self.player1 = player1
        self.player2 = player2
        self.current_player = player1
        self.other_player = player2
        self.empty_text = ""
        self.board_size = board_size
        self.streak_size = streak_size
        self.board = Board(board_size=board_size, streak_size=streak_size)

        self.buttons = [[None for _ in range(board_size)] for _ in range(board_size)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.buttons[i][j] = tk.Button(frame, height=board_size, width=board_size, text=self.empty_text, command=lambda i=i, j=j: self.callback(self.buttons[i][j]))
                self.buttons[i][j].grid(row=i, column=j)

        self.reset_button = tk.Button(text="Reset", command=self.reset)
        self.reset_button.grid(row=board_size)

        self.Q_learn = Q_learn
        if self.Q_learn:
            self.Q = Q
            self.alpha = alpha          # Learning rate
            self.gamma = gamma          # Discount rate
            self.share_Q_with_players()

        self.log_path = "weights_log_boardSize_{boardSize}_streakSize{streak}.csv".format\
                                                                    (boardSize=self.board_size,streak=self.streak_size)
        self.episode_count = 0
        '''
        with open(self.log_path, "r",newline='') as logFile:
            fieldnames = ['episode', 'density', 'linear', 'nonlinear', 'interaction', 'blocking']
            logger = csv.DictWriter(logFile, fieldnames=fieldnames)
            logger.writeheader()
        '''

    @property
    def Q_learn(self):
        if self._Q_learn is not None:
            return self._Q_learn
        if isinstance(self.player1, QPlayer) or isinstance(self.player2, QPlayer):
            return True

    @Q_learn.setter
    def Q_learn(self, _Q_learn):
        self._Q_learn = _Q_learn

    def share_Q_with_players(self):             # The action value table Q is shared with the QPlayers to help them make their move decisions
        if isinstance(self.player1, QPlayer):
            self.player1.Q = self.Q
        if isinstance(self.player2, QPlayer):
            self.player2.Q = self.Q

    def callback(self, button):
        if self.board.over():
            pass                # Do nothing if the game is already over
        else:
            if isinstance(self.current_player, HumanPlayer) and isinstance(self.other_player, HumanPlayer):
                if self.empty(button):
                    move = self.get_move(button)
                    self.handle_move(move)
            elif isinstance(self.current_player, HumanPlayer) and isinstance(self.other_player, ComputerPlayer):
                computer_player = self.other_player
                if self.empty(button):
                    human_move = self.get_move(button)
                    self.handle_move(human_move)
                    if not self.board.over():               # Trigger the computer's next move
                        computer_move = computer_player.get_move(self.board)
                        self.handle_move(computer_move)

    def empty(self, button):
        return button["text"] == self.empty_text

    def get_move(self, button):
        info = button.grid_info()
        move = (int(info["row"]), int(info["column"]))                # Get move coordinates from the button's metadata
        return move

    def handle_move(self, move):
        #if self.Q_learn:
        #    self.learn_Q(move)
        i, j = move         # Get row and column number of the corresponding button
        #self.buttons[i][j].configure(text=self.current_player.mark)     # Change the label on the button to the current player's mark
        self.board.place_mark(move, self.current_player.mark)           # Update the board
        if self.board.over():
            self.declare_outcome()
        ### Osher : no need to switch players in our scenario, each turn includes both players
        ###else:
        ###    self.switch_players()

    def declare_outcome(self):
        if self.board.winner() is None:
            print("Cat's game.")
        else:
            print(("The game is over. The player with mark {mark} won!".format(mark=self.current_player.mark)))
            ###print(self.board.grid)
            self.episode_count += 1
            with open(self.log_path ,"a",newline='') as logFile:
                fieldnames = ['episode', 'density','linear','nonlinear','interaction','blocking']
                logger= csv.DictWriter(logFile, fieldnames=fieldnames)
                row_dict = dict(self.player1.qLearningAgent.getWeights())
                row_dict['episode'] = self.episode_count
                logger.writerow(row_dict)



    def reset(self):
        print("Resetting...")
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.buttons[i][j].configure(text=self.empty_text)
        self.board = Board(board_size=self.board_size, streak_size=self.streak_size)
        self.current_player = self.player1
        self.other_player = self.player2
        # np.random.seed(seed=0)      # Set the random seed to zero to see the Q-learning 'in action' or for debugging purposes
        self.play()

    def switch_players(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
            self.other_player = self.player1
        else:
            self.current_player = self.player1
            self.other_player = self.player2

    def play(self):
        while not self.board.over():
            self.play_turn()
        '''
        if isinstance(self.player1, HumanPlayer) and isinstance(self.player2, HumanPlayer):
            pass        # For human vs. human, play relies on the callback from button presses
        elif isinstance(self.player1, HumanPlayer) and isinstance(self.player2, ComputerPlayer):
            pass
        elif isinstance(self.player1, ComputerPlayer) and isinstance(self.player2, HumanPlayer):
            first_computer_move = self.player1.get_move(self.board)      # If player 1 is a computer, it needs to be triggered to make the first move.
            self.handle_move(first_computer_move)
        elif isinstance(self.player1, ComputerPlayer) and isinstance(self.player2, ComputerPlayer):
            while not self.board.over():        # Make the two computer players play against each other without button presses
                self.play_turn()
        '''
    def getReward(self, nextState)->float:
        ### Osher: check
        if nextState.winner() == "X":
            return 1.0
        elif nextState.winner() == "O":
            return -1.0
        else:
            return 0.0

    def play_turn(self):
        ''' X moves, Y moves and then update (learn)'''
        ###TODO:FIX THIS CODE UGLINESS!
        if self.board.over():
            return
        state = copy.deepcopy(self.board)
        X_move = self.player1.get_move(self.board)
        self.handle_move(X_move)
        if not(self.board.over()):
            self.switch_players()
            O_move = self.player2.get_move(self.board)
            self.handle_move(O_move)
            self.switch_players()  # back to X
        next_state = self.board
        reward = self.getReward(self.board)
        self.player1.qLearningAgent.update(state, X_move, next_state, reward)

    '''
    def learn_Q(self, move):                        # If Q-learning is toggled on, "learn_Q" should be called after receiving a move from an instance of Player and before implementing the move (using Board's "place_mark" method)
        state_key = QPlayer.make_and_maybe_add_key(self.board, self.current_player.mark, self.Q)
        next_board = self.board.get_next_board(move, self.current_player.mark)
        reward = next_board.give_reward()
        next_state_key = QPlayer.make_and_maybe_add_key(next_board, self.other_player.mark, self.Q)
        if next_board.over():
            expected = reward
        else:
            next_Qs = self.Q[next_state_key]             # The Q values represent the expected future reward for player X for each available move in the next state (after the move has been made)
            if self.current_player.mark == "X":
                expected = reward + (self.gamma * min(next_Qs.values()))        # If the current player is X, the next player is O, and the move with the minimum Q value should be chosen according to our "sign convention"
            elif self.current_player.mark == "O":
                expected = reward + (self.gamma * max(next_Qs.values()))        # If the current player is O, the next player is X, and the move with the maximum Q vlue should be chosen
        change = self.alpha * (expected - self.Q[state_key][move])
        self.Q[state_key][move] += change
    '''


class Board:
    def __init__(self, board_size=3, streak_size=3):
        assert board_size>=streak_size
        self.board_size = board_size
        self.grid = np.ones((board_size, board_size)) * np.nan
        self.streak_size = streak_size

    def winner(self):
        rows, cols, diag, cross_diag = self.get_rows_cols_streaks()
        lanes = np.concatenate((rows, cols, diag, cross_diag))      # A "lane" is defined as a row, column, diagonal, or cross-diagonal
        any_lane = lambda x: any([np.array_equal(lane, x) for lane in lanes])   # Returns true if any lane is equal to the input argument "x"
        if any_lane(np.ones(self.streak_size)):
            return "X"
        elif any_lane(np.zeros(self.streak_size)):
            return "O"

    def get_rows_cols_streaks(self):
        # TODO: we might need to convert values to 1,0 and np.nan
        board_mat = np.array(copy.deepcopy(self.grid))
        board_rows = []
        board_cols = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (j+self.streak_size<self.board_size+1):
                    row = []
                    for k in range(self.streak_size):
                        row.append(board_mat[i,j+k])
                    board_rows.append(row)
                    #print(j+self.streak_size)
                    #board_rows.append(board_mat[i, j:(j+self.streak_size])
                    #board_rows.append(row)
                if (i+self.streak_size < self.board_size+1):
                    col = []
                    for k in range(self.streak_size):
                        col.append(board_mat[i + k, j])
                    board_cols.append(col)

        # diagonals
        board_diag = []
        board_cross_diag = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + self.streak_size < self.board_size + 1) and (j + self.streak_size < self.board_size + 1):
                    diag = []
                    for k in range(self.streak_size):
                        diag.append(board_mat[i+k][j+k])
                    board_diag.append(diag)
                if (i+self.streak_size < self.board_size+1) and (self.board_size-j-self.streak_size >= 0):
                    cross_diag = []
                    for k in range(self.streak_size):
                        cross_diag.append(board_mat[i+k][self.board_size-1-j-k])
                    board_cross_diag.append(cross_diag)

        return board_rows,board_cols,board_diag,board_cross_diag

    def over(self):             # The game is over if there is a winner or if no squares remain empty (cat's game)
        return (not np.any(np.isnan(self.grid))) or (self.winner() is not None)

    def place_mark(self, move, mark):       # Place a mark on the board
        num = Board.mark2num(mark)
        self.grid[tuple(move)] = num

    @staticmethod
    def mark2num(mark):         # Convert's a player's mark to a number to be inserted in the Numpy array representing the board. The mark must be either "X" or "O".
        d = {"X": 1, "O": 0, np.nan: np.nan}
        return d[mark]


    def available_moves(self):
        return [(i,j) for i in range(self.board_size) for j in range(self.board_size) if np.isnan(self.grid[i][j])]


    @staticmethod
    def available_moves_static(board):
        # Returns a list of INDICES (i,j)
        if board.over():
            return []
        else:
            return [(i, j) for i in range(board.board_size) for j in range(board.board_size) if np.isnan(board.grid[i][j])]

    def get_next_board(self, move, mark):
        next_board = copy.deepcopy(self)
        next_board.place_mark(move, mark)
        return next_board

    def make_key(self, mark):          # For Q-learning, returns a 10-character string representing the state of the board and the player whose turn it is
        ###Osher: this should be replaced with features!
        fill_value = 3
        filled_grid = copy.deepcopy(self.grid)
        np.place(filled_grid, np.isnan(filled_grid), fill_value)
        return "".join(map(str, (list(map(int, filled_grid.flatten()))))) + mark

    def give_reward(self):                          # Assign a reward for the player with mark X in the current board position.
        if self.over():
            if self.winner() is not None:
                if self.winner() == "X":
                    return 1.0                      # Player X won -> positive reward
                elif self.winner() == "O":
                    return -1.0                     # Player O won -> negative reward
            else:
                return 0.5                          # A smaller positive reward for cat's game
        else:
            return 0.0                              # No reward if the game is not yet finished


class Player(object):
    def __init__(self, mark):
        self.mark = mark

    @property
    def opponent_mark(self):
        if self.mark == 'X':
            return 'O'
        elif self.mark == 'O':
            return 'X'
        else:
            print("The player's mark must be either 'X' or 'O'.")

class HumanPlayer(Player):
    pass

class ComputerPlayer(Player):
    pass

class RandomPlayer(ComputerPlayer):
    def __init__(self,mark):
        super(ComputerPlayer, self).__init__(mark=mark)

    @staticmethod
    def get_move(board):
        moves = board.available_moves()
        if moves:   # If "moves" is not an empty list (as it would be if cat's game were reached)
            return moves[np.random.choice(len(moves))]    # Apply random selection to the index, as otherwise it will be seen as a 2D array

class THandPlayer(ComputerPlayer):
    def __init__(self, mark):
        super(THandPlayer, self).__init__(mark=mark)

    def get_move(self, board):
        moves = board.available_moves()
        if moves:
            for move in moves:
                if THandPlayer.next_move_winner(board, move, self.mark):
                    return move
                elif THandPlayer.next_move_winner(board, move, self.opponent_mark):
                    return move
            else:
                return RandomPlayer.get_move(board)

    @staticmethod
    def next_move_winner(board, move, mark):
        return board.get_next_board(move, mark).winner() == mark


class QPlayer(ComputerPlayer):
    def __init__(self, mark='X', Q={}, epsilon=0.0, discount=0.9, learningRate=0.01):
        super(QPlayer, self).__init__(mark=mark)
        self.Q = Q
        self.epsilon = epsilon ###epsilon - change later from hardcoding the value
        ### Osher: Board.available_moves() might not be a pointer to a function
        actionFn = lambda state: Board.available_moves_static(state) ###lambda board: board.available_moves()
        self.agentOpts = {'actionFn': actionFn, 'epsilon': epsilon, \
                                      'gamma': discount, 'alpha': learningRate}
        self.qLearningAgent = ApproximateQAgent(FeatureExtractor(),**self.agentOpts)

    def get_move(self, board):
        action = self.qLearningAgent.getAction(board)
        return action


        '''
        if np.random.uniform() < self.epsilon:              # With probability epsilon, choose a move at random ("epsilon-greedy" exploration)
            return RandomPlayer.get_move(board)
        else:




            
            state_key = QPlayer.make_and_maybe_add_key(board, self.mark, self.Q)
            Qs = self.Q[state_key]

            if self.mark == "X":
                return QPlayer.stochastic_argminmax(Qs, max)
            elif self.mark == "O":
                return QPlayer.stochastic_argminmax(Qs, min)

    @staticmethod
    def make_and_maybe_add_key(board, mark, Q):     # Make a dictionary key for the current state (board + player turn) and if Q does not yet have it, add it to Q
        default_Qvalue = 1.0       # Encourages exploration
        state_key = board.make_key(mark)
        if Q.get(state_key) is None:
            moves = board.available_moves()
            Q[state_key] = {move: default_Qvalue for move in moves}    # The available moves in each state are initially given a default value of zero
        return state_key

    @staticmethod
    def stochastic_argminmax(Qs, min_or_max):       # Determines either the argmin or argmax of the array Qs such that if there are 'ties', one is chosen at random
        min_or_maxQ = min_or_max(list(Qs.values()))
        if list(Qs.values()).count(min_or_maxQ) > 1:      # If there is more than one move corresponding to the maximum Q-value, choose one at random
            best_options = [move for move in list(Qs.keys()) if Qs[move] == min_or_maxQ]
            move = best_options[np.random.choice(len(best_options))]
        else:
            move = min_or_max(Qs, key=Qs.get)
        return move
    '''

WIN_SCORE = 10
INFINITY_O = 100
INFINITY_X = 100

class FeatureExtractor:
    def __init__(self, density_radius=2, exp=1, o_weight=0.5):
        """
        :param density_radius: what "radius" (in Manhattan distance) should we consider for density feature
        :param exp: parameter creates the non-linearity (i.e., 2 --> squared)
        :param o_weight says how much weight to give for blocking O paths
        """
        self.o_weight = o_weight
        self.density_radius = density_radius
        self.exp = exp


    '''Input: board is the current board state from which we extract features'''
    def extractFeatures(self, board:Board, index, player="X"):
        """
        :param board: the state
        :param index: the action (next move)
        :param player:
        :return:
        """
        N = board.board_size
        board_mat = board.grid

        assert len(index) == 2

        board_matrix = copy.deepcopy(board_mat)
        #self.convert_matrix_xo(board_matrix) ### Osher - we may need some conversion
        # compute path scores
        score_matrix = [[{} for x in range(N)] for y in range(N)]
        score_matrix = np.array(score_matrix)

        paths_data = copy.deepcopy(board_matrix)
        x_turn = True
        o_turn = False
        if player == 'O':
            x_turn = False
            o_turn = True

        row, col = index
        square_mark = board_mat[row][col]

        square_features_scores = util.Counter() # Previous:self.get_new_square_feat_dict()
        density_score = self.densityFeature(board_matrix, row, col, N)
        square_features_scores["density"] = density_score
        linear, nonlinear, interaction, blocking = self.calcNotDensityFeats(board, paths_data,
                                                                            row, col, x_turn, o_turn)
        square_features_scores["linear"] = linear
        square_features_scores["nonlinear"] = nonlinear
        square_features_scores["interaction"] = interaction
        square_features_scores["blocking"] = blocking


        #todo: remove this variable later
        stale_features = util.Counter()
        stale_features["density"] = density_score
        stale_features["linear"] = linear
        #stale_features["nonlinear"] = nonlinear ### Worse results than with first two features
        stale_features["interaction"] = interaction ### Worse results than with first two features

        # TODO: ### CHANGE THIS BACK TO ALL FEATURES!
        return stale_features
        #return square_features_scores

        #active_squares = []  # Empty squares
        ''' Calculate features for each square in the board'''
        '''
        for r in range(N):
            for c in range(N):
                square_val = board.mark2num(board_matrix[r][c])
                if np.isnan(square_val):  # calculate features only for empty squares
                    square_features_scores = self.get_new_square_feat_dict()
                    density_score = self.densityFeature(board_matrix, r, c, N)
                    square_features_scores["density"] = density_score
                    linear, nonlinear, interaction, blocking = self.calcNotDensityFeats(board, paths_data,
                                                                                        r, c, x_turn, o_turn)
                    square_features_scores["linear"] = linear
                    square_features_scores["nonlinear"] = nonlinear
                    square_features_scores["interaction"] = interaction
                    square_features_scores["blocking"] = blocking
                    #update score_matrix
                    score_matrix[r][c] = square_features_scores
                else:
                    #square already has a value
                    score_matrix[r][c] = {"key":"square_is_taken"}
                return score_matrix
        '''


    def densityFeature(self, board_mat:np.array, row, col, N):
        if (row<0 | row>N | col<0 | col>N):
            raise("Row or column index is out of range")

        board_mat = np.array(copy.deepcopy(board_mat))
        #score_board = np.zeros([N][N])  # score board stores density scores for each square
        density_adjacentSquare_score = 1/8
        density_radiusSquare_score = 1/16
        '''
        for row in range(N):
            for col in range(N):
                # for each square in the matrix...
                if np.isnan(board[row][col]): #(Square is empty - explore radius of 2
        '''
        square_density_score = 0.0

        # Make sure indices are within range
        row_index_begin = max(row - 1, 0)
        col_index_begin = max(col - 1, 0)
        row_index_end = min(row + 2, N) # Slices indices are Exclusive, so we use N (instead of N-1) as upper limit
        col_index_end = min(col + 2, N)

        # First, calculate the score of adjacent squares
        adjacent_Xs = board_mat[row_index_begin:row_index_end, col_index_begin:col_index_end]
        adjacent_Xs_count = np.sum(adjacent_Xs == 1)
        # Multiply the number of adjacent Xs by the score factor
        square_density_score += adjacent_Xs_count * density_adjacentSquare_score

        # Count the number of Xs in larger radius from the square
        # We will later subtract the number of adjacent Xs
        row_index_begin = max(row - self.density_radius, 0)
        col_index_begin = max(col - self.density_radius, 0)
        row_index_end = min(row + self.density_radius + 1, N)
        col_index_end = min(col + self.density_radius + 1, N)

        radius_Xs = board_mat[row_index_begin:row_index_end, col_index_begin:col_index_end]
        radius_Xs_count = np.sum(radius_Xs == 1)
        # Subtract the number of adjacent Xs
        radius_Xs_count = radius_Xs_count - adjacent_Xs_count
        # Factorize the number of radius-Xs and add them to the square's score
        square_density_score += radius_Xs_count*density_radiusSquare_score

        return square_density_score

    def calcNotDensityFeats(self, board:Board, paths_data:list, r, c, x_turn, o_turn, player = "X"):
        streak_size = board.streak_size
        board_matrix = copy.deepcopy(board.grid)
        o_weight = self.o_weight
        paths_data = copy.deepcopy(board_matrix)

        # Calculate open paths for X (human) player
        (all_x_features_but_blocking, open_paths_data_x, max_path_x) = \
                                    self.compute_open_paths_data_interaction(r, c, board_matrix, streak_size,
                                                                             player_turn=x_turn)
        #x_paths = self.compute_open_paths_data_interaction(r, c, board_matrix, player_turn=x_turn, exp=exp)

       #Extract feature scores from l_nl_inter_feat_scores_x
        linear_score = all_x_features_but_blocking["linear"]
        nonlinear_score = all_x_features_but_blocking["nonlinear"]
        interaction_score = all_x_features_but_blocking["interaction"]

        # Calculate blocking scores
        blocking_score_x = 0.0
        x_paths_data = []
        for path in open_paths_data_x:
           x_paths_data.append(path[2])
        ### Osher: why do we need paths_data??
        ###paths_data[r][c] = copy.deepcopy(x_paths_data)

       # Calculate open paths for O (opponent) player
        (all_o_features_but_blocking, open_paths_data_o, max_path_o) = \
                            self.compute_open_paths_data_interaction(r, c, board_matrix, streak_size,
                                                                     player='O', player_turn=o_turn)
       #o_paths = self.compute_open_paths_data_interaction(r, c, board_matrix, player='O', player_turn=o_turn)
        blocking_score_o = 0.0


        # Calculate blocking score
        #TODO: remove this line print(board_matrix)
        if (max_path_x == streak_size) & x_turn:
            # Winning move for X
            blocking_score_x = WIN_SCORE
        elif (max_path_o == streak_size) & o_turn:
            blocking_score_o = WIN_SCORE
        elif (max_path_x == (streak_size - 1)) & x_turn:  # give score for blocking O
            # square_score_o = INFINITY_O
            blocking_score_x = 1#INFINITY_O
        elif (max_path_o == (streak_size - 1)) & o_turn:  # give score for blocking X
           # square_score_x = INFINITY_O
            blocking_score_o = 1#+= INFINITY_O

        if o_weight == 0.5:
            blocking_score = blocking_score_x + blocking_score_o
           # if x_turn:
           #     square_score = square_score_x
           # else:
           #     square_score = square_score_o
        elif o_weight == 0:
            blocking_score = blocking_score_x  # o blindness for x player disregard O
        elif o_weight == 1.0:
           ### Osher: in this case, shouldn't we ignore the blocking_score_x?
            blocking_score = blocking_score_x  # o blindness - just use for score how good it would be to block x

        if  blocking_score > WIN_SCORE:
            blocking_score = WIN_SCORE
        # features_scores now holds the scores of all features except the density feature
        return linear_score, nonlinear_score, interaction_score, blocking_score


    def compute_open_paths_data_interaction(self, row:int, col:int, board_mat:np.array,
                                            streak_size:int, player='X',player_turn=True):
        '''
        :param self:
        :param row:
        :param col:
        :param board:
        :param player:
        :param player_turn:
        :return:
        '''
        tmp_score_dict = {
                            "linear": 0.0,
                            "nonlinear": 0.0,
                            "interaction": 0.0,
                            }
        exp = self.exp
        player_val = 1
        other_player_val = 0
        other_player = 'O'
        if player == 'O':
            other_player = 'X'
            player_val = 0
            other_player_val = 1

        max_length_path = 0
        threshold = 0

        open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)

        # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
        for i in range(streak_size):
            r = row - i
            c = col - i
            if (r > len(board_mat) - 1) | (r < 0) | (c > len(board_mat) - 1) | (c < 0):
                continue
            blocked = False  # indicates whether the current way is blocked
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < streak_size) & (square_row < len(board_mat)) & (square_row >= 0) & (
                    square_col < len(board_mat)) & (square_col >= 0):
                if board_mat[square_row][square_col] == other_player_val:
                    blocked = True
                elif board_mat[square_row][square_col] == player_val:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)):
                    # Square is empty, add it to empty_squares only only if it's NOT the current square in play
                    empty_squares.append([square_row, square_col])
                path.append([square_row, square_col])
                square_row += 1
                square_col += 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & (
                    (path_x_count > threshold) | ((player_turn) & (path_x_count + 1) > threshold)):
                # If a path is blocked by the opponent, we disregard it
                # add the path if it's not blocked and if there is already at least one X on it
                if player_turn:
                    # The player draw X in the current square, therefore the path has 1 more X in it
                    open_paths_data.append((path_x_count + 1, empty_squares, path))
                    if (path_x_count + 1) > max_length_path:
                        # update longest path of Xs
                        max_length_path = path_x_count + 1
                elif path_x_count > threshold:
                    # Opponent's turn - places an O in the current square.
                    open_paths_data.append((path_x_count, empty_squares, path))

        # check left-down diagonal
        for i in range(streak_size):
            r = row - i
            c = col + i
            if (r > len(board_mat) - 1) | (r < 0) | (c > len(board_mat) - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < streak_size) & (square_row < len(board_mat)) & (square_row >= 0) & (
                    square_col < len(board_mat)) & (square_col >= 0):
                if board_mat[square_row][square_col] == other_player_val:
                    blocked = True
                elif board_mat[square_row][square_col] == player_val:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])
                path.append([square_row, square_col])
                square_row += 1
                square_col -= 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & ((path_x_count > threshold) | ((
                                                                                                     player_turn) & path_x_count + 1 > threshold)):  # add the path if it's not blocked and if there is already at least one X on it
                if player_turn:
                    open_paths_data.append((path_x_count + 1, empty_squares, path))
                    if (path_x_count + 1) > max_length_path:
                        max_length_path = path_x_count + 1
                elif (path_x_count > threshold):
                    open_paths_data.append((path_x_count, empty_squares, path))

        # check vertical
        for i in range(streak_size):
            r = row - i
            c = col
            if (r > len(board_mat) - 1) | (r < 0) | (c > len(board_mat) - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < streak_size) & (square_row < len(board_mat)) & (square_row >= 0) & (
                    square_col < len(board_mat)) & (square_col >= 0):
                if board_mat[square_row][square_col] == other_player_val:
                    blocked = True
                elif board_mat[square_row][square_col] == player_val:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])

                path.append([square_row, square_col])
                square_row += 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & ((path_x_count > threshold) | ((
                                                                                                     player_turn) & path_x_count + 1 > threshold)):  # add the path if it's not blocked and if there is already at least one X on it
                if player_turn:
                    open_paths_data.append((path_x_count + 1, empty_squares, path))
                    if (path_x_count + 1) > max_length_path:
                        max_length_path = path_x_count + 1
                elif (path_x_count > threshold):
                    open_paths_data.append((path_x_count, empty_squares, path))

        # check horizontal
        for i in range(streak_size):
            r = row
            c = col - i
            if (r > len(board_mat) - 1) | (r < 0) | (c > len(board_mat) - 1) | (c < 0):
                continue
            blocked = False
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < streak_size) & (square_row < len(board_mat)) & (square_row >= 0) & (
                    square_col < len(board_mat)) & (square_col >= 0):
                if board_mat[square_row][square_col] == other_player_val:
                    blocked = True
                elif board_mat[square_row][square_col] == player_val:
                    path_x_count += 1
                elif ((square_col != col) | (square_row != row)):
                    empty_squares.append([square_row, square_col])

                path.append([square_row, square_col])
                square_col += 1
                path_length += 1

            if (path_length == streak_size) & (not blocked) & ((path_x_count > threshold) | ((
                                                                                                     player_turn) & path_x_count + 1 > threshold)):  # add the path if it's not blocked and if there is already at least one X on it
                if player_turn:
                    open_paths_data.append((path_x_count + 1, empty_squares, path))
                    if (path_x_count + 1) > max_length_path:
                        max_length_path = path_x_count + 1
                elif (path_x_count > threshold):
                    open_paths_data.append((path_x_count, empty_squares, path))

        # compute the linear, nonlinear and interactions scores for the cell based on the potential paths

        for i in range(len(open_paths_data)):
            p1 = open_paths_data[i]
            if streak_size == p1[0]:
                # current player has one - update all of the features scores
                for feature in tmp_score_dict:
                    tmp_score_dict[feature] = WIN_SCORE  ### Check with Ofra
                winning_move = True
                break  # Highest score achieved
            else:
                tmp_score_dict["linear"] += p1[0]
                tmp_score_dict["nonlinear"] += 1.0 / math.pow((streak_size - p1[0]), exp)  # score for individual path

                # Calculate interaction-feature score
                for j in range(i + 1, len(open_paths_data)):
                    p2 = open_paths_data[j]
                    if self.check_path_overlap(p2[2], p1[2]):  # pi[2] = a list describing the open path
                        if (
                        not (self.check_path_overlap(p1[1], p2[1]))):  # interaction score if the paths don't overlap
                            # pi[1] = array of empty positions in paths
                            numenator = 0.0 + p1[0] * p2[0]
                            denom = ((streak_size - 1) * (streak_size - 1)) - (p1[0] * p2[0])
                            if denom == 0:
                                # current player wins - O can't block both threats
                                #  - we already updated all features scores in this case
                                tmp_score_dict["interaction"] = WIN_SCORE
                            else:
                                tmp_score_dict["interaction"] += math.pow(numenator / denom, exp)

        return (tmp_score_dict, open_paths_data, max_length_path)

    '''
    checks if the two paths can be blocked by a shared cell
    '''
    @staticmethod
    def check_path_overlap(empty1, empty2):
        for square in empty1:
            if square in empty2:
                return True
        return False

    @staticmethod
    def get_new_square_feat_dict():
        '''
        :return: a new features scores dictionary, later used for each square
        '''
        square_dict = {
            "density": 0.0,
            "linear": 0.0,
            "nonlinear": 0.0,
            "interaction": 0.0,
            "blocking": 0.0
        }
        return square_dict
