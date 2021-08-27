import random
import utils
from reconchess import *
import torch
import reconchess.utilities as rbc_utils
from RBCExample import RBCExample

class AlphaRBC(Player):
    def __init__(self, lookback = 3):
        self.board_history = [] # will be the one-hot encoding so that we don't have to reparse the baord
        self.example_history = []
        self.board = None
        self.color = None

        self.sense_coord = None
        self.sense_board = torch.zeros(8,8)

        self.lookback = lookback

        self.sense_history = []

        self.is_start = True

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

        sense_square = random.choice(sense_actions)

        choice_row = int(sense_square / 8)
        choice_col = sense_square % 8

        self.sense_coord = (choice_row, choice_col)
        self.sense_history.append(self.sense_coord)

        return sense_square

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.board.turn = self.color # set color 
        self.board.clear_stack() # remove stack because we keep playing same color

        # Convert board to a proper state to feed to the network
        board_str = self.board.epd().split(" ", 1)[0]

        # Populate actions
        actions_strs = []
        for action in move_actions:
            temp_board = self.board.copy(stack = False) # copy current board
            temp_board.push(action) # push the action

            actions_strs.append(temp_board.epd().split(" ", 1)[0])

        # Repeat starting board to start history
        if self.is_start:
            self.example_history = [RBCExample(board = board_str, coord = self.sense_coord) for _ in range(self.lookback)]
            self.is_start = False
        # Add to examples
        example = RBCExample(board = board_str,
                            coord = self.sense_coord,
                            actions = actions_strs,
                            prev_examples=self.example_history[-self.lookback:])
        example.convert_to_tensor()
        self.example_history.append(example)

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