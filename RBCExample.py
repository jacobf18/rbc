import numpy as np
import torch
from typing import List
import utils

class RBCExample():
    def __init__(self, board: str, coord: List[tuple[int, int]], actions: List[str] = [], prev_examples: List = []):
        self.board = board
        self.coord = coord
        self.actions = actions
        self.tensor = None
        self.board_tensor = None
        self.lookback = len(prev_examples)
        self.prev_examples = prev_examples

        self.game_result = 0

    def convert_to_tensor(self):
        # Converts an example (board string, sense coordinate, action board strings) to tensor
        # Output has shape (actions [1-n], channels [13], lookback [1-m], 8, 8)
        num_channels = 13 # 12 pieces + 1 sense channels
        num_actions = len(self.actions)

        self.tensor = torch.zeros(num_actions, num_channels, self.lookback + 2, 8, 8)

        # Populate current board
        current_board = torch.from_numpy(utils.one_hot_encoding(self.board))
        self.tensor[:,:num_channels - 1,self.lookback] = current_board.unsqueeze(0).repeat(num_actions,1,1,1)

        # Populate what we can see
        sense_board = utils.convert_sense_to_tensor(self.coord[0], self.coord[1])
        self.tensor[:,-1,-2] = sense_board.unsqueeze(0).repeat(num_actions,1,1)

        # Populate board tensor
        self.board_tensor = self.tensor[0, :, self.lookback: self.lookback + 1, :, :]

        # Populate history (all channels)
        for i, example in reversed(list(enumerate(self.prev_examples))):
            if example.board_tensor == None:
                # create tensor for history if it does not already exist
                example.board_tensor = torch.zeros(num_channels, 1, 8, 8)
                board = torch.from_numpy(utils.one_hot_encoding(example.board))
                example.board_tensor[:-1] = board.unsqueeze(1)
                example.board_tensor[-1] = utils.convert_sense_to_tensor(example.coord[0], example.coord[1]).unsqueeze(0)

            self.tensor[:,:,i:i+1] = example.board_tensor.repeat(num_actions,1,1,1,1)
        
        # Populate actions
        for i, action in enumerate(self.actions):
            move_one_hot = torch.from_numpy(utils.one_hot_encoding(action))
            self.tensor[i,:num_channels - 1,self.lookback + 1:self.lookback + 2,:,:] = torch.unsqueeze(move_one_hot, 1)