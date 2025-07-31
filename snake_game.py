import time
import pygame
from pygame.locals import *  # type: ignore
import random
import numpy as np

SIZE = 20
SCREEN_SIZE = 720

class Snake:
    def __init__(self, parent_screen,length) -> None:
        self.length = length
        self.parent_screen = parent_screen
        self.x = [SIZE]*length
        self.y = [SIZE]*length
        self.block = pygame.image.load('resources/block.jpg').convert()
        self.direction = "down"

    def draw(self):
        self.parent_screen.fill((92,92,92))
        for i in range(self.length):
            self.parent_screen.blit(self.block,dest=(self.x[i],self.y[i]))
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
        for i in range(self.length-1,0,-1) :
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
        
        if self.x[0] < 0:
            self.x[0] = SCREEN_SIZE - SIZE
        if self.x[0] >= SCREEN_SIZE:
            self.x[0] = 0
        if self.y[0] < 0:
            self.y[0] = SCREEN_SIZE - SIZE
        if self.y[0] >= SCREEN_SIZE:
            self.y[0] = SIZE

        self.draw()

    


            
class Apple:
    def __init__(self, parent_scrren) -> None:
        self.image = pygame.image.load('resources/apple.jpg').convert() 
        self.parent_screen = parent_scrren
        self.x = SIZE*3
        self.y = SIZE*3

    def draw(self):
        self.parent_screen.blit(self.image, dest= (self.x,self.y))
        pygame.display.flip()

    def move(self):
        grid_size = SCREEN_SIZE // SIZE
        self.x = random.randint(0, grid_size-1) * SIZE
        self.y = random.randint(0, grid_size-1) * SIZE

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.surface = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
        self.surface.fill(color=(92,192,92))
        self.reset()
        self.reward = 0
        # FIXED: Removed incomplete line "self.snake"
    
    def collision(self, x1, y1, x2, y2) -> bool:
        if (x1 >= x2 and x1 < x2+ SIZE) and (y1 >= y2 and y1 < y2 + SIZE):
            return True
        else:
            return False
        
    def disp_score(self) :
        font = pygame.font.SysFont('arial',30)
        score = font.render(f"score:{self.snake.length-1}",True,(255,255,255))
        self.surface.blit(score,(600,10))
        pygame.display.flip()

    def play(self, action=None):
        self.frame_iteration += 1
        
        # ADDED: Handle action from agent
        if action is not None:
            self.move_snake(action)
            
        self.snake.walk()
        self.apple.draw()
        self.reward = 0
        self.game_over  = False

        if self.collision(self.snake.x[0],self.snake.y[0],self.apple.x,self.apple.y):
            self.apple.move()
            self.inc_len()

        # Fixed: Check collision with all body segments starting from index 1
        # (since index 0 is the head, we check from index 1 onwards)
        for i in range(1, self.snake.length):
            if self.collision(self.snake.x[0],self.snake.y[0],self.snake.x[i],self.snake.y[i]) or self.frame_iteration>100*self.snake.length:
                self.game_over = True
                self.reward = -100
                return self.reward, self.game_over, self.snake.length-1
        return self.reward, self.game_over, self.snake.length-1
    
    def move_snake(self, action):
        # ADDED: Convert action array to direction
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
        self.reward += 10

    def reset(self):
        self.snake = Snake(self.surface,1)
        self.apple = Apple(self.surface)
        self.snake.draw()
        self.apple.draw()
        self.frame_iteration = 0

    def state(self):
        x_head, y_head = self.snake.x[0], self.snake.y[0]

        danger_straight = self.is_collision("straight")
        danger_right = self.is_collision("right")
        danger_left = self.is_collision("left")

        dir_left = self.snake.direction == "left"
        dir_up = self.snake.direction == "up"
        dir_down = self.snake.direction == "down"
        dir_right = self.snake.direction == "right"

        apple_left = self.apple.x < x_head
        apple_down = self.apple.y > y_head
        apple_up = self.apple.y < y_head
        apple_right = self.apple.x > x_head


        state =[
            danger_straight,
            danger_right,
            danger_left,


            dir_down,
            dir_left,
            dir_right,
            dir_up,

            apple_down,
            apple_left,
            apple_up,
            apple_right
        ]
        return np.array(state,dtype=int)


    def is_collision(self,action):
        clock_wise =["up","right","down","left"]
        idx = clock_wise.index(self.snake.direction)
        
        # FIXED: Proper handling of all action types
        if action == "straight":
            new_dir = clock_wise[idx]
            new_idx = idx
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
        if new_dir == 'right':
            next_x += SIZE
        if new_dir == 'up' :
            next_y -= SIZE
        if new_dir == 'down':
            next_y += SIZE

        if next_x < 0:
            next_x = SCREEN_SIZE - SIZE
        if next_x >= SCREEN_SIZE:
            next_x = 0
        if next_y < 0:
            next_y = SCREEN_SIZE - SIZE
        if next_y >= SCREEN_SIZE:
            next_y = 0
        
        # Check collision with body
        for i in range(1, self.snake.length):
            if next_x == self.snake.x[i] and next_y == self.snake.y[i]:
                return True
        return False
        

    def run(self):
        running = True
        while running :
            for event in pygame.event.get():
                if event.type == QUIT:
                    running= False
                
            self.play()
            time.sleep(0.2)