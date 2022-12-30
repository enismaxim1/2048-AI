from typing import List
import matplotlib.pyplot as plt
import numpy as np
from random import randrange, random
from abc import ABC, abstractmethod
from game2048 import *
import math

from util import hash_board

class Agent(ABC):
    DIRECTIONS = [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]

    @abstractmethod
    def next_move(self, game: Game2048):
        pass

    def _type(self):
        return self.__class__.__name__

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

    def heuristic(self, game):
        heuristic = 0
        row = game.game_board[0]
        for i in range(0, len(row) - 1):
            if row[i] == 0:
                continue
            heuristic += math.log2(row[i]) * (5-i)
        heuristic += (game.BOARD_SIZE ** 2 - game.get_num_tiles()) * math.log2(game.game_board[0][0] + 1)
        return heuristic


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

    def heuristic(self, game):
        heuristic = 0
        row = game.game_board[0]
        for i in range(0, len(row) - 1):
            if row[i] == 0:
                continue
            heuristic += math.log2(row[i]) * (5-i)
        heuristic += (game.BOARD_SIZE ** 2 - game.get_num_tiles()) * math.log2(game.game_board[0][0] + 1)
        return heuristic


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

    def heuristic(self, game):
        heuristic = 0
        row = game.game_board[0]
        for i in range(0, len(row) - 1):
            if row[i] == 0:
                continue
            heuristic += math.log2(row[i]) * (5-i)
        heuristic += (game.BOARD_SIZE ** 2 - game.get_num_tiles()) * math.log2(game.game_board[0][0] + 1)
        return heuristic

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

    class TreeNode():

        def __init__(self, game = Game2048, max_node = True, score = 0, num_simulations = 0, children = {}):
            self.game = game
            self.max_node = max_node
            self.score = score
            self.num_simulations = num_simulations
            self.children = children

        def add_children(self):
            new_children = {}
            if self.max_node:
                for direction in self.game.valid_directions:
                    if not self.game.is_valid_move(direction):
                        continue
                    new_game = deepcopy(self.game)
                    new_game.move(direction, add_tile = False)
                    # a child is a node along with an edge weight
                    new_children[hash_board(new_game.game_board)] = (type(self)(new_game, False), 1, direction)
            else:
                for cell_x, cell_y in self.game.find_empty_cells():
                    for tile_num in [2,4]:
                        new_game = deepcopy(self.game)
                        new_game.game_board[cell_x][cell_y] = tile_num
                        edge_weight = new_game.PROB_2 if tile_num == 2 else 1 - new_game.PROB_2
                        # a child is a node along with an edge weight   
                        new_children[hash_board(new_game.game_board)] = (type(self)(new_game, True), edge_weight, (cell_x, cell_y))
            self.children = new_children
        
        def __str__(self):
            return str(self.game)

    def __init__(self, explore_prob = 0.2):
        self.search_agent = SearchAgent(1)
        self.random_agent = RandomAgent()
        self.explore_prob = explore_prob
        self.game_tree = None
        self.memo = {}


    def next_move(self, game):
        if game.get_max_num() < 256:
            return self.search_agent.next_move(game)

        if not self.game_tree:
            self.game_tree = self.TreeNode(game)
        else:
            self.game_tree, _, _ = self.game_tree.children[hash_board(game.game_board)]

        for _ in range(4):
            self.run_mcts_iteration(self.game_tree)

        max_num_simulations, max_direction = 0, None
        for child, _, direction, in self.game_tree.children.values():
            if child.num_simulations >= max_num_simulations:
                max_num_simulations, max_direction = child.num_simulations, direction
        self.game_tree = child
        return max_direction
        
    
    def run_mcts_iteration(self, root: TreeNode):
        path = self.select(root)
        leaf = path[-1]
        # add child nodes to leaf, if nonterminal
        if not leaf.game.lose():
            leaf.add_children()
            # run simulations on all children of node
            for child, _, _ in leaf.children.values():
                result = self.simulate(child)
                # backpropogate results to every node in path
                self.backpropagate(path, result)
        print("iter")
    def select(self, root):

        def find_best_child(root):
            max_avg_score, max_child = 0, None
            for child, _, _ in root.children.values():
                avg_score = 0 if not child.num_simulations else child.score / child.num_simulations
                if avg_score >= max_avg_score:
                    max_avg_score, max_child = avg_score, child
            return max_child

        # select until we reach a leaf
        path = [root]
        node = root
        while node.children:
            node = find_best_child(node)
            path.append(node)
        return path
    
    def simulate(self, root):
        is_max_node = root.max_node
        game_copy = deepcopy(root.game)
        if not is_max_node:
            game_copy.add_tile_to_board()

        hashed_board = hash_board(root.game.game_board)
        if hashed_board in self.memo:
            return self.memo[hashed_board]

        while not game_copy.lose():
            game_copy.move(self.search_agent.next_move(game_copy))
        
        result = math.log2(game_copy.get_max_num())
        self.memo[hashed_board] = result
        return result

    def backpropagate(self, path: List[TreeNode], result):
        for node in reversed(path):
            node.num_simulations += 1
            if node.max_node:
                node.score += result
            else:
                for child, edge_weight, _ in node.children.values():
                    avg_score = 0 if not child.num_simulations else child.score / child.num_simulations
                    node.score += avg_score * edge_weight





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
                print(agent._type(), max_num)
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

    while True:
        move = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
 
        move = agent.next_move(game)
        if isinstance(move, Direction) and game.is_valid_move(move):
            if game.move(move):
                draw_game(game, window_size, screen)

def main():
    # agent = TestPerformance([SmarterAgent(), SearchAgent(), MonteCarloAgent()])
    # print(agent.evaluate_performance())
    play_ai_game(MonteCarloTreeSearchAgent())

if __name__ == "__main__":
    main()


