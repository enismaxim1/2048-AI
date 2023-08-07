from time import time
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, random, choice
from abc import ABC, abstractmethod
from game2048 import *
from monte_carlo_tree import *
import math

from util import *

class Agent(ABC):
    DIRECTIONS = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]

    @abstractmethod
    def next_move(self, game: Game2048):
        pass

    def _type(self):
        return self.__class__.__name__

    def heuristic(self, game):
        heuristic = 0
        row = game.game_board[0]
        for i in range(0, len(row) - 1):
            if row[i] == 0:
                continue
            heuristic += math.log2(row[i]) * (5-i)
        heuristic += (game.BOARD_SIZE ** 2 - game.get_num_tiles()) * math.log2(game.game_board[0][0] + 1)
        return heuristic

class RandomAgent(Agent):

    def next_move(self, game):
        move = None
        while not move or not game.is_valid_move(move):
            move = self.DIRECTIONS[randrange(4)]
        return move

class SmarterAgent(Agent):

    def next_move(self, game):
        for direction in self.DIRECTIONS:
            if game.is_valid_move(direction):
                return direction

class SearchAgent(Agent):

    def __init__(self, default_search_depth = 3):
        super()
        self.default_search_depth = default_search_depth

    def next_move(self, game):
        _, move = self.multi_level_heuristic(game, 2)
        return move

    def multi_level_heuristic(self, game: Game2048, depth):
        if depth == 0:
            return 0, None
        
        new_game = deepcopy(game)

        max_heuristic, best_move = 0, None
        for direction in self.DIRECTIONS:
            if not new_game.is_valid_move(direction):
                continue
            curr_game = deepcopy(new_game)
            curr_game.move(direction)
            prev_heuristic = self.heuristic(curr_game)
            heuristic, _ = self.multi_level_heuristic(curr_game, depth - 1)
            if heuristic + prev_heuristic > max_heuristic:
                    max_heuristic, best_move = heuristic + prev_heuristic, direction
        return max_heuristic, best_move

class AveragingSearchAgent(Agent):

    def __init__(self, default_search_depth = 3):
        super()
        self.default_search_depth = default_search_depth

    def next_move(self, game):
        _, move = self.multi_level_heuristic(game, self.default_search_depth)
        return move

    def multi_level_heuristic(self, game: Game2048, depth):
        if depth == 0:
            return 0, None
        
        new_game = deepcopy(game)

        num_avg = 1
        max_heuristic, best_move = 0, None
        for direction in self.DIRECTIONS:
            if not new_game.is_valid_move(direction):
                continue
            heuristics = []
            for _ in range(num_avg):
                curr_game = deepcopy(new_game)
                curr_game.move(direction)
                curr_heuristic = self.heuristic(curr_game)
                heuristic, _ = self.multi_level_heuristic(curr_game, depth - 1)
                heuristics.append(curr_heuristic + heuristic)

            avg_heuristic = sum(heuristics) / num_avg
            if avg_heuristic > max_heuristic:
                    max_heuristic, best_move = avg_heuristic, direction
        return max_heuristic, best_move

class MonteCarloAgent(Agent):

    policy = SmarterAgent()

    def next_move(self, game):
        num_avg = 10
        best_average, best_move = 0, None
        for direction in self.DIRECTIONS:
            if not game.is_valid_move(direction):
                continue
            scores = []
            for _ in range(num_avg):
                new_game = deepcopy(game)
                new_game.move(direction)
                num_steps, max_steps = 0, 7
                while not new_game.lose() and num_steps < max_steps:
                    new_game.move(self.policy.next_move(new_game))
                    num_steps += 1
                scores.append(self.heuristic(new_game))
            avg_score = sum(scores) / num_avg
            if avg_score > best_average:
                best_average, best_move = avg_score, direction
        return best_move

class MonteCarloTreeSearchAgent(Agent):

    def __init__(self, num_iterations = 5):
        self.search_agent = SearchAgent(1)
        self.random_agent = RandomAgent()
        self.game_tree = None
        self.num_iterations = num_iterations
        self.memo = {}

    def next_move(self, game):

        if not self.game_tree:
            self.game_tree = MaxNode(game = game)
            self.game_tree.score += self.simulate(self.game_tree)
            self.game_tree.num_simulations += 1
        else:
            self.game_tree = self.game_tree.children[hash_board(game.game_board)]


        for i in range(self.num_iterations):
            self.run_mcts_iteration(self.game_tree)

        max_num_simulations, max_direction = 0, None
        last_child = None
        for child in self.game_tree.children.values():
            if not game.is_valid_move(child.direction):
                continue
            if child.num_simulations > max_num_simulations:
                max_num_simulations, max_direction = child.num_simulations, child.direction
                last_child = child
        self.game_tree = last_child
        return max_direction
        
    
    def run_mcts_iteration(self, root):
        path = self.select(root)
        leaf = path[-1]
        # add child nodes to leaf, if nonterminal
        if not leaf.game.lose():
            leaf.add_children()
            # run simulations on all children of node
            for expectation_child in leaf.children.values():
                expectation_child.add_children()
                path.append(expectation_child)
                for child in expectation_child.children.values():
                    result = self.simulate(child)
                    path.append(child)
                    # backpropagate results to every node in path
                    self.backpropagate(path, result)
                    path.pop()
                path.pop()
        else:
            raise Exception(f"Game leaf {leaf} is unexpectedly losing")

    def select(self, root):

        def find_best_child(root):
            max_policy, max_child = 0, None
            C = 0.2
            for child in root.children.values():
                avg_score = child.avg_score() if isinstance(child, ExpectationNode) else 0.01 * random()
                policy =  avg_score + C * math.sqrt(math.log2(child.parent.num_simulations) / child.num_simulations)
                if policy >= max_policy:
                    max_policy, max_child = policy, child
            return max_child

        # select until we reach a leaf
        path = [root]
        node = root
        while node.children:
            node = find_best_child(node)
            path.append(node)
        return path

    def random_path(self, root):
        path = [root]
        node = root
        while node.children:
            node = choice([child for child in node.children.values()])
            path.append(node)
        return path
    
    def simulate(self, root):
        game_copy = deepcopy(root.game)
        max_num_moves = 2
        num_moves = 0
        while not game_copy.lose() and num_moves < max_num_moves:
            game_copy.move(self.search_agent.next_move(game_copy))
            num_moves += 1

        heuristic = self.heuristic(game_copy)
        scaled = heuristic / ((math.log2(game_copy.tile_sum)) * 26 + 12)
        if scaled > 1:
            raise Exception(f"Scaled heuristic value {heuristic} exceeds 1")
        return scaled

    def backpropagate(self, path, result):
        for node in reversed(path):
            node.num_simulations += 1
            if isinstance(node, MaxNode):
                node.score += result
                if (avg_node_score := node.score/node.num_simulations > 1):
                    raise Exception(f"Average node score {avg_node_score} exceeds 1")


class TestPerformance:
    NUM_TRIALS = 50

    def __init__(self, agents):
        self.agents = agents

    def evaluate_performance(self):
        agent_scores, agent_maxes = [], []
        for agent in self.agents:
            scores, maxes = [], []
            for _ in range(self.NUM_TRIALS):
                game = Game2048()
                while not game.lose():
                    next_move = agent.next_move(game)
                    game.move(next_move)
                score, max_num = game.score, int(math.log2(game.get_max_num()))
                scores.append(score)
                maxes.append(max_num)
            agent_scores.append(scores)
            agent_maxes.append(maxes)

        self.plot_results(agent_maxes, self.agents)

        return sum(scores) / self.NUM_TRIALS, sum(maxes) / self.NUM_TRIALS, self.percent_winning(maxes)

    def percent_winning(self, maxes):
        return len([max for max in maxes if max >= 11]) / len(maxes)

    def plot_results(self, agent_results, agents):
        for index, results in enumerate(agent_results):
            score_range = range(min(results), max(results) + 1)
            score_freqs = [results.count(score) for score in score_range]
            scores = list(score_range)
            plt.plot(scores, score_freqs)
        plt.legend([agent._type() for agent in agents])
        plt.show()

def play_ai_game(agent):
    game = Game2048()
    pygame.init()
    window_size = 600
    size = window_size, window_size

    screen = pygame.display.set_mode(size)
    draw_game(game, window_size, screen)
    num_moves = 0
    start_time = time()
    while True:
        move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        if not game.lose():
            move = agent.next_move(game)
            num_moves += 1
            if isinstance(move, Direction) and game.is_valid_move(move):
                if game.move(move):
                    draw_game(game, window_size, screen, avg_move_time = (time() - start_time) / num_moves)

def main():
    # agent = TestPerformance([AveragingSearchAgent(), MonteCarloAgent()])
    # print(agent.evaluate_performance())
    play_ai_game(MonteCarloTreeSearchAgent())

if __name__ == "__main__":
    main()


