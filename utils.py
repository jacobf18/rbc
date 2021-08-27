import numpy as np
import torch
from typing import List
from functools import lru_cache

@lru_cache
def one_hot_encoding(string_rep: str) -> np.array:
    # Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -
    mat = np.array(make_matrix(string_rep))

    # Pieces as a char list
    piece_types = list('rnbqkpRNBQKP')

    # Get one-hot encoding (# pieces, 8 height, 8 width)
    one_hot = np.zeros([12,8,8], dtype=np.int32)
    for i, piece in enumerate(piece_types):
        one_hot[i] = (mat == piece).astype(np.int32)

    return one_hot

@lru_cache
def make_matrix(pieces):
    foo = []
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append('.')
            else:
                foo2.append(thing)
        foo.append(foo2)
    return foo

def convert_sense_to_tensor(choice_row: int, choice_col: int):
    sense_board = torch.zeros(8,8)
    sense_board[choice_row, choice_col] = 1
    
    # Corners
    # Top left
    if choice_row != 0 and choice_col != 0:
        sense_board[choice_row - 1, choice_col - 1] = 1
    # Top right
    if choice_row != 0 and choice_col != 7:
        sense_board[choice_row - 1, choice_col + 1] = 1
    # Bottom left
    if choice_row != 7 and choice_col != 0:
        sense_board[choice_row + 1, choice_col - 1] = 1
    # Bottom right
    if choice_row != 7 and choice_col != 7:
        sense_board[choice_row + 1, choice_col + 1] = 1
    
    # Sides
    # Top
    if choice_row != 0:
        sense_board[choice_row - 1, choice_col] = 1
    # Left
    if choice_col != 0:
        sense_board[choice_row, choice_col - 1] = 1
    # Right
    if choice_col != 7:
        sense_board[choice_row, choice_col + 1] = 1
    # Bottom
    if choice_row != 7:
        sense_board[choice_row + 1, choice_col] = 1
    
    return sense_board