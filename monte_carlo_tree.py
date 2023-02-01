from game2048 import *
from util import hash_board

class TreeNode():

    def __init__(self, game: Game2048 = None, score = 0, num_simulations = 0, children = {}, parent = None):
        self.game = game
        self.score = score
        self.num_simulations = num_simulations
        self.children = children
        self.parent = parent


class ExpectationNode(TreeNode):

    def __init__(self, direction = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction = direction
        
    def add_children(self):
        if self.children:
            raise Exception("Node already has children")
        new_children = {}
        for cell_x, cell_y in self.game.find_empty_cells():
            for tile_num in [2,4]:
                new_game = deepcopy(self.game)
                num_empty = len(new_game.find_empty_cells())
                new_game.add_tile_to_board((cell_x, cell_y), tile_num)
                edge_weight = new_game.PROB_2 / num_empty if tile_num == 2 else (1 - new_game.PROB_2) / num_empty
                # a child is a node along with an edge weight   
                new_children[hash_board(new_game.game_board)] = MaxNode(game = new_game, edge_weight = edge_weight, cell_pos = (cell_x, cell_y), parent = self)
        self.children = new_children
    
    def avg_score(self):
        score = 0
        prob_sum = 0
        for child in self.children.values():
            prob_sum += child.edge_weight
            score += child.edge_weight * child.score / child.num_simulations 
            if child.score / child.num_simulations > 1:
                raise Exception(f"FUCK! {child.score} {child.num_simulations}")
        if prob_sum > 1.1:
            raise Exception(f"prob sum is {prob_sum}")
        if score > 1:
            raise Exception("FUCK!")
        return score

    def __str__(self):
        string =  f"num simulations: {self.num_simulations}\ngame:\n{str(self.game)}\ndirection: {self.direction}"
        return string

class MaxNode(TreeNode):
    def __init__(self, edge_weight = 0, cell_pos = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weight = edge_weight
        self.cell_pos = cell_pos

    def add_children(self):
        if self.children:
            raise Exception("Node already has children")
        new_children = {}
        for direction in self.game.valid_directions:
            if not self.game.is_valid_move(direction):
                raise Exception("WTF")
            new_game = deepcopy(self.game)
            new_game.move(direction, add_tile = False)
            # a child is a node along with an edge weight
            new_children[hash_board(new_game.game_board)] = ExpectationNode(game = new_game, direction = direction, parent = self)
        self.children = new_children

    def __str__(self):
        string =  f"num simulations: {self.num_simulations}\ngame:\n{str(self.game)}\nedge weight: {self.edge_weight}\ncell pos = {self.cell_pos}"
        return string