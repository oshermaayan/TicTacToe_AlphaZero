import game
import policy_value_net_numpy
import pickle
import operator
import json
import numpy as np

#TODO: delete the following paths -lines
#board_file_path = "board_file_6.txt"
#model_file = 'best_policy_8_8_5.model'#'best_policy_6_6_4.model'

def file_to_board(board_file_path,rows_num, cols_num, X="X",O="O",empty="_",splitter="\t"):
    '''
    :param board_file_path:
    :param X: X representation in the file
    :param O:
    :param empty:
    :return: A game.Board() object, filled with the file's setting
    '''
    board = game.Board(width=cols_num, height=rows_num)
    board.init_board()
    with open (board_file_path, "r") as board_file:
        rows = board_file.read().splitlines()
        assert len(rows) == rows_num
        for i, row in enumerate(rows):
            cols = row.split(splitter)
            assert len(cols) == cols_num
            for j, col in enumerate(cols):
                move = i*cols_num + j
                if col == X:
                    board.do_move_manual(move, 1)
                elif col == O:
                    board.do_move_manual(move, 2)

    return board

def json_file_to_board(json_path,rows_num, cols_num, matrix_key, X=1,O=2,empty=0):
    '''
    :param board_file_path: path to file with array-like matrix
    :param X: X representation in the matrix
    :param matrix_key: key (String) to the matrix value inside the json file; e.g. "position"
    :param O:
    :param empty:
    :return: 1. A game.Board() object, filled with the file's setting
    '''
    with open (json_path, "r") as json_file:
        data = json.load(json_file)
        matrix = data[matrix_key]

    matrix_np = np.array(matrix)
    assert matrix_np.shape == (rows_num, cols_num)

    #Initialize a Board object
    board = game.Board(width=cols_num, height=rows_num)
    board.init_board()

    for i in range(rows_num):
        for j in range(cols_num):
            move = i * cols_num + j
            if matrix_np[i][j] == X:
                board.do_move_manual(move, 1)
            elif matrix_np[i][j] == O:
                board.do_move_manual(move, 2)
            else:
                continue

    return board

def extract_marked_location(json_path,rows_num, cols_num, matrix_key, X=1,O=2,empty=0):
    '''
    :param board_file_path: path to file with array-like matrix
    :param X: X representation in the matrix
    :param matrix_key: key (String) to the matrix value inside the json file; e.g. "position"
    :return: 1. A list containing locations of marked Xs in the format [(row1,col1),(row2,col2)...]
             2. A similar list for marked Os
    '''
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        matrix = data[matrix_key]

    matrix_np = np.array(matrix)
    assert matrix_np.shape == (rows_num, cols_num)

    X_locations = []
    O_locations = []

    for i in range(rows_num):
        for j in range(cols_num):
            if matrix_np[i][j] == X:
                X_locations.append((i, j))
            elif matrix_np[i][j] == O:
                O_locations.append((i, j))
            else:
                continue

    return X_locations, O_locations

def extract_scores(DQN_results, board):
    actions_score = {}
    for move, probability in DQN_results:
        location = board.move_to_location(move)
        (row, col) = (location[0], location[1])
        actions_score[(row, col)] = probability

    return actions_score

def create_probabilites_matrix(actions_dict, X_locations, O_locations, board_size):
    '''

    :param actions_dict: dictionary of (row,col)->probability
    :param original_board: the original board
    :return: a numpy matrix, in which each square contains:
                1. The original mark (X or O) if the square isn't empty
                2. The probability to choose this action - otherwise
    '''
    result_mat = np.zeros((board_size,board_size))

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

#Main
json_path = "json_6x6.json"
model_file = 'best_policy_6_6_4.model2'
rows_num = 6
cols_num = 6

# Set up board
board = json_file_to_board(json_path, matrix_key= "position", rows_num=rows_num, cols_num=cols_num)

# Extract X and O locations in the original board
X_locations, O_locations = extract_marked_location(json_path, matrix_key= "position", rows_num=rows_num, cols_num=cols_num)

#Create an evaluator for each board and evaluate current board
policy_param = pickle.load(open(model_file, 'rb'),encoding='bytes')
evaluator = policy_value_net_numpy.PolicyValueNetNumpy(rows_num, cols_num, policy_param)

zipped_res, board_score = evaluator.policy_value_fn(board) # Calculated for new_board.current_player!
actions_scores = extract_scores(zipped_res, board)

heat_map = create_probabilites_matrix(actions_scores, X_locations, O_locations, board_size=rows_num)


'''
best_move_index = max(actions_scores.items(), key=operator.itemgetter(1))[0]

best_move_index = (best_move_index[0] + 1, best_move_index[1] + 1) #Show "Matlab" index

print(actions_scores)
print("\n"*2)
print("Best move is {best_move_index}".format(best_move_index=best_move_index))
'''
#TODO: delete the following lines later (they were used for sanity-check tests)
#win, winner = new_board.has_a_winner(original_code=False)
#print(win, winner)

