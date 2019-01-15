import pickle as pickle    # cPickle is for Python 2.x only; in Python 3, simply "import pickle" and the accelerated version will be used automatically if available
import numpy as np
import math
from numpy.core.multiarray import ndarray
import copy
#from Q_Learning_Tic_Tac_Toe import Board
import util

WIN_SCORE = 10
INFINITY_O = 100
INFINITY_X = 100

class FeatureExtractor:
    def __init__(self, streak_size, density_radius=2, exp=1, o_weight=0.5):
        """
        :param density_radius: what "radius" (in Manhattan distance) should we consider for density feature
        :param exp: parameter creates the non-linearity (i.e., 2 --> squared)
        :param o_weight says how much weight to give for blocking O paths
        """
        self.o_weight = o_weight
        self.density_radius = density_radius
        self.exp = exp
        self.streak_size = streak_size


    '''Input: board is the current board state from which we extract features'''
    def extractFeatures(self, board, index, player="X"):
        """
        :param board: the state - numpy array
        :param index: the action (next move)
        :param player:
        :return:
        """
        N = board.shape[0] #Assuming an NxN board
        board_mat = board

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

        square_features_scores = util.Counter() # Previous:self.get_new_square_feat_dict()
        density_score = self.densityFeature(board_matrix, row, col, N)
        square_features_scores["density"] = density_score
        linear, nonlinear, interaction, blocking = self.calcNotDensityFeats(board, paths_data,
                                                                            row, col, x_turn, o_turn,
                                                                            streak_size=self.streak_size)
        square_features_scores["linear"] = linear
        square_features_scores["nonlinear"] = nonlinear
        square_features_scores["interaction"] = interaction
        square_features_scores["blocking"] = blocking

        return square_features_scores

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


    def densityFeature(self, board_mat:np.array, row, col, N, X=1, O=2):
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
        adjacent_Xs_count = np.sum(adjacent_Xs == X)
        # Multiply the number of adjacent Xs by the score factor
        square_density_score += adjacent_Xs_count * density_adjacentSquare_score

        # Count the number of Xs in larger radius from the square
        # We will later subtract the number of adjacent Xs
        row_index_begin = max(row - self.density_radius, 0)
        col_index_begin = max(col - self.density_radius, 0)
        row_index_end = min(row + self.density_radius + 1, N)
        col_index_end = min(col + self.density_radius + 1, N)

        radius_Xs = board_mat[row_index_begin:row_index_end, col_index_begin:col_index_end]
        radius_Xs_count = np.sum(radius_Xs == X)
        # Subtract the number of adjacent Xs
        radius_Xs_count = radius_Xs_count - adjacent_Xs_count
        # Factorize the number of radius-Xs and add them to the square's score
        square_density_score += radius_Xs_count*density_radiusSquare_score

        return square_density_score

    def calcNotDensityFeats(self, board, paths_data:list, r, c, x_turn, o_turn, streak_size,
                                                 X=1, O=2):

        '''

        :param board: A numpy-array board, representing the state
        :param paths_data:
        :param r: the row of the square to calculate features-scores for
        :param c: the column of the square to calculate features-scores for
        :param x_turn:
        :param o_turn:
        :param streak_size: the streak size to win
        :param X: The value representing X on the board
        :param O: The value representing O on the board
        :return:
        '''
        streak_size = streak_size
        board_matrix = copy.deepcopy(board)
        o_weight = self.o_weight
        paths_data = copy.deepcopy(board_matrix)

        # Calculate open paths for X (human) player
        (all_x_features_but_blocking, open_paths_data_x, max_path_x) = \
                                    self.compute_open_paths_data_interaction(r, c, board_matrix, streak_size,
                                                                             player_turn=x_turn,
                                                                             player = X,
                                                                             other_player= O)
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
                                                                     player=O,
                                                                     other_player=X,
                                                                        player_turn=o_turn)
       #o_paths = self.compute_open_paths_data_interaction(r, c, board_matrix, player='O', player_turn=o_turn)
        blocking_score_o = 0.0


        # Calculate blocking score
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
                                            streak_size:int, player =1 , other_player = 2, player_turn=True):
        '''
        :param self:
        :param row:
        :param col:
        :param board:
        :param player: VALUE of player on the board
        :param player_turn:
        :return:
        '''
        tmp_score_dict = {
                            "linear": 0.0,
                            "nonlinear": 0.0,
                            "interaction": 0.0,
                            }
        exp = self.exp

        max_length_path = 0
        threshold = 0

        open_paths_data = []  # this list will hold information on all the potential paths, each path will be represented by a pair (length and empty squares, which will be used to check overlap)

        # check right-down diagonal (there is a more efficient way to look at all the paths, but it was easier for me to debug when separating them :)
        for i in range(streak_size):
            r = row - i
            c = col - i
            if (r > len(board_mat)- 1) | (r < 0) | (c > len(board_mat) - 1) | (c < 0):
                continue
            blocked = False # indicates whether the current way is blocked
            path_length = 0
            path_x_count = 0
            empty_squares = []
            path = []
            square_row = r
            square_col = c
            while (not blocked) & (path_length < streak_size) & (square_row < len(board_mat)) & (square_row >= 0) & (
                    square_col < len(board_mat)) & (square_col >= 0):
                if board_mat[square_row][square_col] == other_player:
                    blocked = True
                elif board_mat[square_row][square_col] == player:
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
                if board_mat[square_row][square_col] == other_player:
                    blocked = True
                elif board_mat[square_row][square_col] == player:
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
                if board_mat[square_row][square_col] == other_player:
                    blocked = True
                elif board_mat[square_row][square_col] == player:
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
                if board_mat[square_row][square_col] == other_player:
                    blocked = True
                elif board_mat[square_row][square_col] == player:
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
                    tmp_score_dict[feature] = WIN_SCORE ### Check with Ofra
                winning_move = True
                break # Highest score achieved
            else:
                tmp_score_dict["linear"] += p1[0]
                tmp_score_dict["nonlinear"] += 1.0 / math.pow((streak_size - p1[0]), exp)  # score for individual path

                # Calculate interaction-feature score
                for j in range(i + 1, len(open_paths_data)):
                    p2 = open_paths_data[j]
                    if self.check_path_overlap(p2[2], p1[2]):  # pi[2] = a list describing the open path
                        if (not (self.check_path_overlap(p1[1], p2[1]))):  # interaction score if the paths don't overlap
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
