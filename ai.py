import matplotlib.pyplot as plt
import numpy as np
from random import randrange
from abc import ABC, abstractmethod
from game2048 import *
import math

class Agent(ABC):
    DIRECTIONS = [Direction.LEFT, Direction.DOWN, Direction.RIGHT, Direction.UP]

    @abstractmethod
    def next_move(self, game: Game2048):
        pass

class RandomAgent(Agent):

    def next_move(self, game):
        move = None
        while not move or not game.is_valid_move(move):
            move = self.DIRECTIONS[randrange(4)]
        return move

class SmarterAgent(Agent):
    DIRECTIONS = [Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN]

    def next_move(self, game):
        for direction in self.DIRECTIONS:
            if game.is_valid_move(direction):
                return direction


class GreedyAgent(Agent):
    DIRECTIONS = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]
    num_moves = 0

    def next_move(self, game):
        _, moves = self.multi_level_heuristic(game, game.game_board, 5)
        self.num_moves += 1
        return moves[-1]

    def early_heuristic(self, game, board):
        heuristic = 2 * (game.BOARD_SIZE ** 2 - game.get_num_tiles(board))
        if board[0][0] != game.get_max_num(board):
            heuristic = heuristic / 2
        return heuristic

    def heuristic(self, game, board):
        heuristic = 0
        row = board[0]
        for i in range(0, len(row) - 1):
            if row[i] == 0:
                continue
            heuristic += math.log2(row[i]) * (5-i)
        heuristic += (game.BOARD_SIZE ** 2 - game.get_num_tiles(board)) * math.log2(board[0][0] + 1)
        return heuristic


    def multi_level_heuristic(self, game, board, depth):
        if depth == 0:
            return 0, []
        
        max_heuristic, best_moves = 0, []
        for index, direction in enumerate(self.DIRECTIONS):
            new_board, _ = game.collapse_tiles(direction, board)
            if new_board == board:
                continue
            board_heuristic = self.heuristic(game, new_board)
            game.add_tile_to_board(new_board)
            prev_heuristic, prev_moves = self.multi_level_heuristic(game, new_board, depth - 1)
            if prev_heuristic + board_heuristic > max_heuristic:
                max_heuristic, best_moves = prev_heuristic + board_heuristic, prev_moves + [direction]
        return max_heuristic, best_moves


class TestPerformance:
    NUM_TRIALS = 20

    def __init__(self, agent):
        self.agent = agent

    def evaluate_performance(self):
        scores, maxes = [], []
        for _ in range(self.NUM_TRIALS):
            game = Game2048()
            while not game.lose():
                next_move = self.agent.next_move(game)
                game.move(next_move)
            score, max_num = game.score, int(math.log2(game.get_max_num(game.game_board)))
            scores.append(score)
            maxes.append(max_num)
            print(max_num)

        # fig, ax = plt.subplots()
        # bins = np.arange(min(maxes), max(maxes) + 1.5) - 0.5
        # plt.hist(maxes, bins)
        # ax.set_xticks(bins + 0.5)
        return sum(scores) / self.NUM_TRIALS, sum(maxes) / self.NUM_TRIALS, self.percent_winning(maxes)

    def percent_winning(self, maxes):
        return len([max for max in maxes if max >= 11]) / len(maxes)


def play_ai_game(agent):
    game = Game2048()
    pygame.init()
    window_size = 600
    size = window_size, window_size

    screen = pygame.display.set_mode(size)
    draw_game(game, window_size, screen)

    while True:
        move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.KEYDOWN:
                move = agent.next_move(game)
        if isinstance(move, Direction) and game.is_valid_move(move):
            if game.move(move):
                draw_game(game, window_size, screen)

def main():
    # greedy = TestPerformance(GreedyAgent())
    # print(greedy.evaluate_performance())
    play_ai_game(GreedyAgent())

if __name__ == "__main__":
    main()


