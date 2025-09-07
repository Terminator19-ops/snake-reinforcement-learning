import time
import pygame
from pygame.locals import *  # type: ignore
import random
import numpy as np

SIZE = 100  
SCREEN_SIZE = 1100  

# Grid calculation for maximum score
GRID_WIDTH = SCREEN_SIZE // SIZE   # 11 squares
GRID_HEIGHT = SCREEN_SIZE // SIZE  # 11 squares
MAX_SQUARES = GRID_WIDTH * GRID_HEIGHT  # 121 total squares

class Snake:
    def __init__(self, parent_screen, length) -> None:
        self.length = length
        self.parent_screen = parent_screen
        self.x = [SIZE]*length
        self.y = [SIZE]*length
        self.block = pygame.Surface((SIZE, SIZE))
        self.block.fill((255, 255, 0)) # Blue color
        self.direction = "down"

    def draw(self):
        self.parent_screen.fill((92,92,92))
        for i in range(self.length):
            self.parent_screen.blit(self.block, dest=(self.x[i], self.y[i]))
        pygame.display.flip()

    def set_direction(self, direction):
        valid_directions = ["up", "down", "left", "right"]
        if direction not in valid_directions:
            return
        # Prevent reversing direction
        opposite_directions = {
            "up": "down",
            "down": "up", 
            "left": "right",
            "right": "left"
        }
        
        if self.direction != opposite_directions[direction]:
            self.direction = direction

    def walk(self):
        for i in range(self.length-1, 0, -1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        if self.direction == "left":
            self.x[0] -= SIZE
        if self.direction == "right":
            self.x[0] += SIZE
        if self.direction == "up":
            self.y[0] -= SIZE
        if self.direction == "down":
            self.y[0] += SIZE
        
        self.draw()

class Food:  # Renamed from Apple to Food
    def __init__(self, parent_screen) -> None:
        # Try to load apple image, fallback to red colored rectangle
        self.image = pygame.Surface((SIZE, SIZE))
        self.image.fill((255, 0, 0))
        self.parent_screen = parent_screen
        self.x = SIZE*3
        self.y = SIZE*3

    def draw(self):
        self.parent_screen.blit(self.image, dest=(self.x, self.y))
        pygame.display.flip()

    def move(self):
        grid_size = SCREEN_SIZE // SIZE
        self.x = random.randint(0, grid_size-1) * SIZE
        self.y = random.randint(0, grid_size-1) * SIZE

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.surface = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption('Snake AI')
        self.surface.fill(color=(92, 192, 92))
        self.reset()
        self.reward = 0

    def collision(self, x1, y1, x2, y2) -> bool:
        if (x1 >= x2 and x1 < x2 + SIZE) and (y1 >= y2 and y1 < y2 + SIZE):
            return True
        else:
            return False
        
    def disp_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f"score:{self.snake.length-1}", True, (255, 255, 255))
        self.surface.blit(score, (SCREEN_SIZE-150, 10))
        pygame.display.flip()

    def play(self, action=None):
        self.frame_iteration += 1
        
        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Handle action from agent
        if action is not None:
            self.move_snake(action)
            
        self.snake.walk()
        self.food.draw()  # Changed from apple to food
        # Initialize reward
        self.reward = -0.01  # Small negative reward per step to encourage efficiency
        self.game_over = False

        # Snake ate food - significant reward
        if self.collision(self.snake.x[0], self.snake.y[0], self.food.x, self.food.y):
            self.food.move()
            self.inc_len()
            self.reward = 10 + (self.snake.length * 0.1)  # Reward increases with snake length
        
        # Calculate distance to food
        prev_distance = self.prev_food_distance if hasattr(self, 'prev_food_distance') else None
        current_distance = np.sqrt((self.snake.x[0] - self.food.x)**2 + (self.snake.y[0] - self.food.y)**2)
        
        # Reward for getting closer to food
        if prev_distance and current_distance < prev_distance:
            self.reward += 0.1
        elif prev_distance and current_distance > prev_distance:
            self.reward -= 0.1
            
        self.prev_food_distance = current_distance
        
        # Check boundary collision
        if (self.snake.x[0] < 0 or self.snake.x[0] >= SCREEN_SIZE or 
            self.snake.y[0] < 0 or self.snake.y[0] >= SCREEN_SIZE):
            self.game_over = True
            self.reward = -10
            return self.reward, self.game_over, self.snake.length-1

        # Check collision with snake body
        for i in range(1, self.snake.length):
            if self.collision(self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]):
                self.game_over = True
                self.reward = -10
                return self.reward, self.game_over, self.snake.length-1
        
        # Add frame limit to prevent infinite games
        if self.frame_iteration > 100 * len(self.snake.x):
            self.game_over = True
            self.reward = -10
            return self.reward, self.game_over, self.snake.length-1

        return self.reward, self.game_over, self.snake.length-1
    
    def move_snake(self, action):
        # Convert action array to direction
        # action = [straight, right, left]
        clock_wise = ["up", "right", "down", "left"]
        idx = clock_wise.index(self.snake.direction)
        
        if np.array_equal(action, [1, 0, 0]):  # straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        elif np.array_equal(action, [0, 0, 1]):  # left
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            new_dir = clock_wise[idx]  # default to straight
            
        self.snake.set_direction(new_dir)
        
    def inc_len(self):
        self.snake.length += 1
        self.snake.x.append(-1) 
        self.snake.y.append(-1) 

    def reset(self):
        self.snake = Snake(self.surface, 1)
        self.food = Food(self.surface)  # Changed from apple to food
        self.snake.draw()
        self.food.draw()
        self.frame_iteration = 0

    def is_collision(self, action):
        clock_wise = ["up", "right", "down", "left"]
        idx = clock_wise.index(self.snake.direction)
        
        # Handle all action types
        if action == "straight":
            new_dir = clock_wise[idx]
        elif action == "right":
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        elif action == "left":
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]
        else:
            return False
        
        next_x = self.snake.x[0]
        next_y = self.snake.y[0]

        if new_dir == 'left':
            next_x -= SIZE
        elif new_dir == 'right':
            next_x += SIZE
        elif new_dir == 'up':
            next_y -= SIZE
        elif new_dir == 'down':
            next_y += SIZE

        # Check boundary collision
        if next_x < 0 or next_x >= SCREEN_SIZE or next_y < 0 or next_y >= SCREEN_SIZE:
            return True
        
        # Check collision with body
        for i in range(1, self.snake.length):
            if next_x == self.snake.x[i] and next_y == self.snake.y[i]:
                return True
        return False

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                
            self.play()
            time.sleep(0.2)