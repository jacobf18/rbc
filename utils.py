import numpy as np

def make_matrix(pgn):
    foo = []
    pieces = pgn.split(" ", 1)[0]
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
    
    