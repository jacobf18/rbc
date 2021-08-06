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
        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(move_actions, seconds_left)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        # otherwise, just randomly choose a sense action, but don't sense on a square where our pieces are located
        for square, piece in self.board.piece_map().items():
            if piece.color == self.color:
                sense_actions.remove(square)
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add the pieces in the sense result to our board
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # Convert board to a proper state to feed to the network
        one_hot = torch.from_numpy(utils.one_hot_encoding(self.board.epd()))
        
        actions = list(self.board.generate_pseudo_legal_moves())
        print(len(actions))
        # for action in actions:
        #     temp_board = self.board.copy(stack = False)
            # print(temp_board.unicode())
            # temp_board.push(action)
            # move_one_hot = torch.from_numpy(utils.one_hot_encoding(temp_board.epd()))
        
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass