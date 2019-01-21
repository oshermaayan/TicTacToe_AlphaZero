import json
import numpy as np
from FeatureExtractor import FeatureExtractor
import game
import policy_value_net_numpy
import pickle
import operator


def read_board_from_json(json_path, board_key, rows_num, cols_num):
    '''
    :param json_path: path to json file
    :param board_key: key to the board value
    :param rows_num
    :param cols_num
    :return: the board object
    '''
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        board = np.array(data[board_key])
        assert board.shape == (rows_num, cols_num)
    return board

def extract_X_O_locations(board, X=1,O=2):
    '''
    :param board:
    :param X: X's mark in the original board, default is 1
    :param O: O's mark in the original board, default is 2
    :return: two lists, one for X and one for O, containing their locations (row,col) on the board
    '''
    rows_num, cols_num = board.shape
    X_locations = [(row, col) for col in range(cols_num) for row in range(rows_num) if board[row, col] == X]
    O_locations = [(row, col) for col in range(cols_num) for row in range(rows_num) if board[row, col] == O]

    return X_locations, O_locations

def convert_to_AlphaZeroBoard(orig_board, X_locations, O_locations):
    '''
    :param orig_board: the original game board
    :return: an "AlphaZero board"  with the relevant places marked
    '''
    # Initialize a Board object
    rows_num, cols_num = orig_board.shape
    az_board = game.Board(height=rows_num, width=cols_num)
    az_board.init_board()

    #Place marks for X
    for x_row, x_col in X_locations:
        move = x_row*cols_num + x_col
        az_board.do_move_manual(move, 1)
    #Do the same for O
    for o_row, o_col in O_locations:
        move = o_row*cols_num + o_col
        az_board.do_move_manual(move, 2)

    return az_board

def extract_features_scores(board, features_names, featExt):
    '''
    :param board: game's board
    :param features_names: a list of features names
    :param featExt: a FeatureExtractor object
    :return:
    '''
    score_maps = {}
    for feature in features_names:
        score_board = np.zeros(board.shape)
        feature_sum = 0.0 #Later used to normalize scores
        non_zero_feature_vals_index = []
        for row in range(rows_num):
            for col in range (cols_num):
                if board[row, col] == 0:
                    #Calculate features
                    square_feat = featExt.extractFeatures(board=board,index=(row,col))
                    value = square_feat[feature]
                    if value > 0:
                        feature_sum += value
                        non_zero_feature_vals_index.append((row,col))
                        score_board[row, col] = value
                else:
                    # Square is already marked - keep  mark (X/O) from original board
                    score_board[row, col] = board[row, col]

        if feature_sum > 0:
            for index in non_zero_feature_vals_index:
                assert not board[index] # assert square in the original board is    empty
                score_board[index] = score_board[index]/feature_sum
        score_maps[feature] = score_board

    return score_maps

def extract_AlphaZero_scores(DQN_results, board):
    actions_score = {}
    for move, probability in DQN_results:
        location = board.move_to_location(move)
        (row, col) = (location[0], location[1])
        actions_score[(row, col)] = probability

    return actions_score

def create_AlphaZero_prob_matrix(actions_dict, X_locations, O_locations, rows_num, cols_num):
    '''

    :param actions_dict: dictionary of (row,col)->probability
    :param original_board: the original board
    :return: a numpy matrix, in which each square contains:
                1. The original mark (X or O) if the square isn't empty
                2. The probability to choose this action - otherwise
    '''
    result_mat = np.zeros((rows_num,cols_num))

    #Fill X's and O's locations first
    for X_loc_row,X_loc_col in X_locations:
        result_mat[X_loc_row, X_loc_col] = 1
    for O_loc_row, O_loc_col in O_locations:
        result_mat[O_loc_row, O_loc_col] = 2

    # Fill the rest of the squares with probabilites

    for square, prob in actions_dict.items():
        # square is of the form (row,col)
        # Verify square is empty - just for sanity check
        assert result_mat[square[0]][square[1]] == 0
        result_mat[square[0]][square[1]] = prob

    return result_mat

def get_AlphaZero_actions_scores(model_path, az_board):
    '''
    :param model_path: path to trained model file
    :param az_board: alphaZero board
    :return: actions scores - a dict {(row,col)->probability}
    '''
    # Create an evaluator for each board and evaluate current board
    policy_param = pickle.load(open(model_path, 'rb'), encoding='bytes')
    evaluator = policy_value_net_numpy.PolicyValueNetNumpy(rows_num, cols_num, policy_param)

    zipped_res, board_score = evaluator.policy_value_fn(az_board)  # Scores are alculated for new_board.current_player!
    actions_scores = extract_AlphaZero_scores(zipped_res, az_board)
    return actions_scores, board_score

#Main
json_path = "json_6x6.json"
json_board_key = "position"
model_file = 'best_policy_6_6_4.model2'
rows_num = 6
cols_num = 6
streak_size = 4
X_mark = 1
O_mark = 2
features_names = ["density", "linear", "nonlinear", "interaction", "blocking"]

np_board = read_board_from_json(json_path, json_board_key, rows_num, cols_num)
X_locations, O_locations = extract_X_O_locations(np_board, X_mark, O_mark)

#Create score boards for our features
fe = FeatureExtractor(streak_size=streak_size)
features_scores_boards = extract_features_scores(np_board, features_names, fe)

#Create score board for AlphaZero results
az_board = convert_to_AlphaZeroBoard(np_board, X_locations, O_locations)
actions_scores, board_score = get_AlphaZero_actions_scores(model_file, az_board)
alphaZero_score_board = create_AlphaZero_prob_matrix(actions_scores, X_locations, O_locations, rows_num, cols_num)

print("Done")