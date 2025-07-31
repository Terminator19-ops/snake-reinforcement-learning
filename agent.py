import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
from collections import deque
from snake_game import Game 
import torch.optim as optim
import matplotlib.pyplot as plt

MAX_MEM = 100_000
BATCH = 1000
LR = 0.001

class QNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx].item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class AGENT:
    def __init__(self):
        self.ngames = 0
        self.epsilon = max(5, 80 - self.ngames)
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        snake_head_x = game.snake.x[0]
        snake_head_y = game.snake.y[0]
        apple_x = game.apple.x
        apple_y = game.apple.y

        danger_straight = game.is_collision("straight")
        danger_right = game.is_collision("right") 
        danger_left = game.is_collision("left")

        dir_up = game.snake.direction == "up"
        dir_down = game.snake.direction == "down"
        dir_left = game.snake.direction == "left"
        dir_right = game.snake.direction == "right"

        apple_up = apple_y < snake_head_y
        apple_down = apple_y > snake_head_y
        apple_left = apple_x < snake_head_x
        apple_right = apple_x > snake_head_x

        state = [
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

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_on_long_memory(self):
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_on_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.ngames
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        return move 

def plot(scores, mean_scores):
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = AGENT()
    game = Game()
    plt.ion()
    import pygame
    FPS = 50 
    clock = pygame.time.Clock()

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)  # Now an index
        final_move = [0, 0, 0]
        final_move[action] = 1
        reward, done, score = game.play(final_move)
        state_new = agent.get_state(game)
        agent.train_on_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.ngames += 1
            agent.train_on_long_memory()
            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.ngames
            plot_mean_scores.append(mean_score)
            print(f'Game {agent.ngames}, Score {score}, Record: {record}')
            plot(plot_scores, plot_mean_scores)

        clock.tick(FPS)  # Control the frame rate

if __name__ == "__main__":
    train()
