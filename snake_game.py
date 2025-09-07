import time
import pygame
from pygame.locals import *  # type: ignore
import random

SIZE = 20
SCREEN_SIZE = 720

class Snake:
    def __init__(self, parent_screen, length) -> None:
        self.length = length
        self.parent_screen = parent_screen
        self.x = [SIZE]*length
        self.y = [SIZE]*length
        self.block = pygame.image.load('resources/block.jpg').convert()
        self.direction = "down"

    def draw(self):
        # Only draw the snake blocks, don't flip the display here
        for i in range(self.length):
            self.parent_screen.blit(self.block, dest=(self.x[i], self.y[i]))

    def move_up(self):
        self.direction = "up"
    
    def move_down(self):
        self.direction = "down"
    
    def move_left(self):
        self.direction = "left"
    
    def move_right(self):
        self.direction = "right"

    def inc_len(self):
        self.length += 1
        self.x.append(-1) 
        self.y.append(-1)
    
    def walk(self):
        # Move body segments
        for i in range(self.length-1, 0, -1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        # Move head based on direction
        if self.direction == "left":
            self.x[0] -= SIZE
        if self.direction == "right":
            self.x[0] += SIZE
        if self.direction == "up":
            self.y[0] -= SIZE
        if self.direction == "down":
            self.y[0] += SIZE
        
        # Handle screen wrapping
        if self.x[0] < 0:
            self.x[0] = SCREEN_SIZE - SIZE
        if self.x[0] >= SCREEN_SIZE:
            self.x[0] = 0
        if self.y[0] < 0:
            self.y[0] = SCREEN_SIZE - SIZE
        if self.y[0] >= SCREEN_SIZE:
            self.y[0] = 0

class Apple:
    def __init__(self, parent_screen) -> None:
        self.image = pygame.image.load('resources/apple.jpg').convert() 
        self.parent_screen = parent_screen
        self.x = SIZE*3
        self.y = SIZE*3

    def draw(self):
        # Only draw the apple, don't flip the display here
        self.parent_screen.blit(self.image, dest=(self.x, self.y))

    def move(self):
        self.x = random.randint(0, 17)*SIZE
        self.y = random.randint(0, 17)*SIZE

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.surface = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        pygame.display.set_caption("Snake Game")
        self.snake = Snake(self.surface, 1)
        self.apple = Apple(self.surface)
    
    def collision(self, x1, y1, x2, y2) -> bool:
        if (x1 >= x2 and x1 < x2 + SIZE) and (y1 >= y2 and y1 < y2 + SIZE):
            return True
        else:
            return False
        
    def disp_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f"Score: {self.snake.length-1}", True, (255, 255, 255))
        self.surface.blit(score, (600, 10))

    def play(self):
        # Clearing the screen 
        self.surface.fill((92, 92, 92))
        
        # Move snake
        self.snake.walk()
        
        # Check collision with apple
        if self.collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move()
            self.snake.inc_len()
        
        # Check collision with itself
        for i in range(3, self.snake.length):
            if self.collision(self.snake.x[0], self.snake.y[0], self.snake.x[i], self.snake.y[i]):
                return False  # Game over
        
        # Draw everything
        self.snake.draw()
        self.apple.draw()
        self.disp_score()
        
        # Update display only once per frame
        pygame.display.flip()
        return True

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    if event.key == K_UP:
                        self.snake.move_up()
                    if event.key == K_DOWN:
                        self.snake.move_down()
                    if event.key == K_LEFT:
                        self.snake.move_left()
                    if event.key == K_RIGHT:
                        self.snake.move_right()
                elif event.type == QUIT:
                    running = False
            
            if not self.play():  # Game over
                running = False
            
            time.sleep(0.2)
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()