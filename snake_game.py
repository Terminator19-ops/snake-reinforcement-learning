import time
import pygame
from pygame.locals import *  # type: ignore
import random

SIZE = 40
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

    def move_up(self):
        self.direction= "up"
    def move_down(self):
        self.direction= "down"
    def move_left(self):
        self.direction= "left"
    def move_right(self):
        self.direction= "right"

    def inc_len(self):
        self.length += 1
        self.x.append(-1) 
        self.y.append(-1) 
    
    def walk(self):
        for i in range(self.length-1,0,-1) :
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        if self.direction == "left":
            self.x[0] -= SIZE
            self.x[0] += SIZE
        if self.direction == "up":
            self.y[0] -= SIZE
        if self.direction == "down":
            self.y[0] += SIZE
        
        if self.x[0] < 0:
            self.x[0] = SCREEN_SIZE - 40
        if self.x[0] > SCREEN_SIZE:
            self.x[0] = 0
        if self.y[0] < 0:
            self.y[0] = SCREEN_SIZE - SIZE
        if self.y[0] > SCREEN_SIZE:
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
        self.x= random.randint(0,17)*SIZE
        self.y= random.randint(0,17)*SIZE

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.surface = pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
        self.surface.fill(color=(92,192,92))
        self.snake = Snake(self.surface,1)
        self.apple = Apple(self.surface)
        self.snake.draw()
        self.apple.draw()
    
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

    def play(self) :
        self.snake.walk()
        self.apple.draw()

        if self.collision(self.snake.x[0],self.snake.y[0],self.apple.x,self.apple.y):
            self.apple.move()
            self.snake.inc_len()
        self.disp_score()

        for i in range(3,self.snake.length) :
            if self.collision(self.snake.x[0],self.snake.y[0],self.snake.x[i],self.snake.y[i]):
                exit(0)

    def run(self):
        running = True
        while running :
            for event in pygame.event.get():
                if event.type == KEYDOWN :
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
                    running= False
            self.play()
            time.sleep(0.2)


if __name__ == "__main__" :
    game = Game()
    Game().run()

