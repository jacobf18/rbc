import random
import utils
from reconchess import *
from MCTS import MCTS
import torch
import reconchess.utilities as rbc_utils

class AlphaRBC(Player):
    def __init__(self):
        self.board_history = [] # will be the one-hot encoding so that we don't have to reparse the baord
        self.board = None
        self.color = None
        # self.mcts = MCTS()
        self.examples = [1]

        self.sense_board = torch.zeros(8,8)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        # Reset sense_board
        self.sense_board *= 0

        choices = torch.tensor(sense_actions).reshape(8,8)
        sense_square = random.choice(sense_actions)

        choice_row = int(sense_square / 8)
        choice_col = sense_square % 8
        
        # Center square
        self.sense_board[choice_row, choice_col] = 1
        
        # Corners
        # Top left
        if choice_row != 0 and choice_col != 0:
            self.sense_board[choice_row - 1, choice_col - 1] = 1
        # Top right
        if choice_row != 0 and choice_col != 7:
            self.sense_board[choice_row - 1, choice_col + 1] = 1
        # Bottom left
        if choice_row != 7 and choice_col != 0:
            self.sense_board[choice_row + 1, choice_col - 1] = 1
        # Bottom right
        if choice_row != 7 and choice_col != 7:
            self.sense_board[choice_row + 1, choice_col + 1] = 1
        
        # Sides
        # Top
        if choice_row != 0:
            self.sense_board[choice_row - 1, choice_col] = 1
        # Left
        if choice_col != 0:
            self.sense_board[choice_row, choice_col - 1] = 1
        # Right
        if choice_col != 7:
            self.sense_board[choice_row, choice_col + 1] = 1
        # Bottom
        if choice_row != 7:
            self.sense_board[choice_row + 1, choice_col] = 1

        return sense_square

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.board.turn = self.color # set color 
        self.board.clear_stack() # remove stack because we keep playing same color

        # Convert board to a proper state to feed to the network
        num_actions = len(move_actions)
        lookback = 10
        num_channels = 13
        state_action = torch.zeros(num_actions, num_channels, lookback + 2, 8, 8)

        # Populate the current board
        board = torch.from_numpy(utils.one_hot_encoding(self.board.epd()))
        state_action[:,:num_channels - 1,lookback:lookback + 1,:,:] = torch.unsqueeze(board, 1).repeat(num_actions,1,1,1,1)

        # Populate what we can see
        state_action[:,num_channels - 1:num_channels,lookback:lookback+2] = self.sense_board.expand(1,1,1,8,8).repeat(num_actions,1,1,1,1)

        # Populate the history
        history = self.board_history[-lookback:]
        for i in range(len(history)):
            ind = len(history) - i - 1
            state_action[:,:,ind:ind+1] = history[ind].repeat(num_actions,1,1,1,1)
        
        # Add the current board configuration to the history
        self.board_history += [state_action[0, :, lookback: lookback + 1, :, :]]

        # Populate actions
        for i, action in enumerate(move_actions):
            temp_board = self.board.copy(stack = False)
            temp_board.push(action)

            # Convert to one-hot encoding
            move_one_hot = torch.from_numpy(utils.one_hot_encoding(temp_board.epd()))

            # Add each action's one-hot encoding to the state tensor
            state_action[i,:num_channels - 1,lookback + 1:lookback + 2,:,:] = torch.unsqueeze(move_one_hot, 1)

        # TODO: Run through net to get Q value and probs

        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass