
import math


def hash_board(board):
    return hash(tuple([tuple(row) for row in board]))

def sigmoid(val, scale):
    return 1 / (1+math.exp(-val / scale))