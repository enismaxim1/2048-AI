from random import random, randrange
from copy import deepcopy
from enum import Enum
import pygame

class Direction(Enum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


class Game2048:

    BOARD_SIZE = 4
    PROB_2 = .9
    NUM_START_TILES = 2

    def __init__(self):
        self.score = 0
        self.initialize_game_board()
        self.valid_directions = list(Direction)

    def __str__(self):
        return "\n".join(str(row) for row in self.game_board)
    
    def initialize_game_board(self):
        self.game_board = [[0 for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        for _ in range(self.NUM_START_TILES):
            self.add_tile_to_board(self.game_board)

    def find_empty_cells(self, board):
        empty = []
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    empty.append((i, j))
        return empty

    def add_tile_to_board(self, board):
        random_float = random()
        number = None
        if random_float < self.PROB_2:
            number = 2
        else:
            number = 4
        
        empty_cells = self.find_empty_cells(board)
        x_index, y_index = empty_cells[randrange(0, len(empty_cells))]
        board[x_index][y_index] = number
    
    def collapse_tiles(self, direction, board):
        if not isinstance(direction, Direction):
            raise Exception(f"{direction} is not a valid direction.")
        new_board = self.transform_dir(board, direction)
        score = 0
        for i in range(len(board)):
            score += self.collapse_row(new_board[i])
        return self.transform_dir(new_board, direction, reverse = True), score

    def collapse_row(self, row):
        score = 0
        left_index = 0
        right_index = 1
        while right_index < len(row):
            left_val = row[left_index]
            right_val = row[right_index]
            if right_val == 0:
                right_index += 1
                continue
            if left_val == 0:
                row[left_index] = right_val
                row[right_index] = 0
            elif left_val == right_val:
                row[left_index] = 2 * left_val
                row[right_index] = 0
                score += left_val
                left_index += 1
            elif left_val != right_val:
                row[right_index] = 0
                row[left_index + 1] = right_val
                left_index += 1
            right_index += 1 
        return score

    
    def move(self, direction):
        if not self.is_valid_move(direction):
            raise Exception(f"Move {direction} is not valid")
        new_board, score = self.collapse_tiles(direction, self.game_board)
        self.game_board = new_board
        self.add_tile_to_board(self.game_board)
        self.score += score
        self.compute_valid_directions()
        return True

    # Computes the valid directions for the board state.
    def compute_valid_directions(self):
        valid_directions = []
        for direction in Direction:
            new_board, score = self.collapse_tiles(direction, self.game_board)
            if new_board == self.game_board:
                continue
            valid_directions.append(direction)
        self.valid_directions = valid_directions

    def is_valid_move(self, direction):
        return direction in self.valid_directions

    def lose(self):
        return len(self.valid_directions) == 0

    def rotate(self, board, k):

        def rotate_once(arr):
            length, width = len(arr), len(arr[0])
            rotated_arr = [[0 for _ in range(length)] for _ in range(width)]
            for i in range(length):
                for j in range(width):
                    rotated_arr[width - j - 1][i] = arr[i][j]
            return rotated_arr

        k %= 4
        rotated = deepcopy(board)
        for _ in range(k):
            rotated = rotate_once(rotated)
        return rotated

    def transform_dir(self, board, direction, reverse = False):
        if direction == Direction.LEFT:
            return self.rotate(board, 0)
        elif direction == Direction.RIGHT:
            return self.rotate(board, 2)
        elif direction == Direction.UP:
            k = 1 if not reverse else -1
            return self.rotate(board, k)
        elif direction == Direction.DOWN:
            k = -1 if not reverse else 1
            return self.rotate(board, k)
        print(direction)
        if not isinstance(direction, Direction):
            raise Exception("Not a valid direction")
    
    def get_max_num(self, board):
        return max([max(row) for row in board])

    def get_num_tiles(self, board):
        num_nonzero = 0
        for row in board:
            num_nonzero += len([elem for elem in row if elem != 0])
        return num_nonzero

rect_color_map = {0: "gray", 2: "wheat1", 4: "wheat2", 8: "tan1"}


def draw_game(game: Game2048, window_size, screen):
    screen.fill("white")
    margin = window_size / 10
    block_size = (window_size - 2 * margin) / game.BOARD_SIZE
    for i in range(game.BOARD_SIZE):
        for j in range(game.BOARD_SIZE):
            x,y = margin + j * block_size, margin + i * block_size
            rect = pygame.Rect(x, y, block_size, block_size)
            element = game.game_board[i][j]
            if element != 0:
                font = pygame.font.SysFont('arial', int(block_size // 2.5))
                text = font.render(str(element), True, (0, 0, 0))
                text_rect = text.get_rect(center = (x + block_size / 2, y + block_size / 2))
                screen.blit(text, text_rect)
            pygame.draw.rect(screen, (0,0,0), rect, 1)

    pygame.display.update()

def play_user_game():
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
                if event.key == pygame.K_a:
                    move = Direction.LEFT
                if event.key == pygame.K_d:
                    move = Direction.RIGHT
                if event.key == pygame.K_w:
                    move = Direction.UP
                if event.key == pygame.K_s:
                    move = Direction.DOWN
        if isinstance(move, Direction) and game.is_valid_move(move):
            if game.move(move):
                draw_game(game, window_size, screen)


def main():
    play_user_game()

if __name__ == "__main__":
    main()