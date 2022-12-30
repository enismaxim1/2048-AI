

def hash_board(board):
    return hash(tuple([tuple(row) for row in board]))