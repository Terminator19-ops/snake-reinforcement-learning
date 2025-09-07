import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
from collections import deque
from snake_game import Game 
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import pygame
from safetensors.torch import save_file, load_file

# Try to import safetensors for better model saving
# Hyperparameters
MAX_MEM = 100_000  # Reduced to focus on more recent experiences
BATCH = 1000       # Reduced batch size for more stable updates
LR = 0.001         # Increased learning rate for faster convergence early on
GAMMA = 0.98       # Increased discount factor to value future rewards more
EPSILON_START = 100 # Start with full exploration
EPSILON_MIN = 1     # Lower minimum exploration for better exploitation
EPSILON_DECAY = 0.99  # Slightly faster decay to reach exploitation sooner

# Learning rate scheduler parameters
LR_DECAY = 0.9999   # Learning rate decay factor
LR_MIN = 0.00001    # Minimum learning rate

# Calculate maximum possible score with new dimensions
SCREEN_SIZE = 1100  # From snake_game.py
BLOCK_SIZE = 100    # From snake_game.py
GRID_WIDTH = SCREEN_SIZE // BLOCK_SIZE  # 11
GRID_HEIGHT = SCREEN_SIZE // BLOCK_SIZE  # 11
MAX_POSSIBLE_SCORE = (GRID_WIDTH * GRID_HEIGHT) - 1  # 120 (total squares minus 1 for head)

print(f" Grid: {GRID_WIDTH}x{GRID_HEIGHT} = {GRID_WIDTH*GRID_HEIGHT} squares")
print(f" Maximum possible score: {MAX_POSSIBLE_SCORE}")

class QNet(nn.Module):
    def __init__(self, input_size=11, hidden_size=256, output_size=3):
        super().__init__()
        # 1 hidden layer for model to understand complex relations
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def save(self, file_name='model.safetensors'):
        """Save model as safetensors"""
        if not file_name.endswith('.safetensors'):
            file_name += '.safetensors'
        
        save_file(self.state_dict(), file_name)
        print(f"Model saved as: {file_name}")
    
    def load(self, file_name='model.safetensors'):
        """Load model from safetensors"""
        if not file_name.endswith('.safetensors'):
            file_name += '.safetensors'
        
        state_dict = load_file(file_name)
        self.load_state_dict(state_dict)
        print(f"Model loaded from: {file_name}")

    def forward(self, x):
        # passing through the layers
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.initial_lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.train_step_counter = 0
    
    def adjust_learning_rate(self):
        """Decaying learning rate over time"""
        self.train_step_counter += 1
        if self.train_step_counter % 100 == 0:  # update after every 100 steps
            self.lr = max(LR_MIN, self.lr * LR_DECAY)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
    
    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to torch tensors for computation
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # If input is a single sample, add batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Get current Q-value predictions from the model
        pred = self.model(state)
        # Clone predictions to use as targets for loss calculation
        target = pred.clone()

        # Update target Q-values using the Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # If not done, add discounted max Q-value for next state
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # Find the index of the action taken
            if isinstance(action[idx], torch.Tensor):
                action_idx = torch.argmax(action[idx]).item()
            else:
                action_idx = action[idx].item() if hasattr(action[idx], 'item') else action[idx]
            # Update the target for the action taken
            target[idx][action_idx] = Q_new

        # Zero gradients, compute loss, backpropagate, and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        # Optionally adjust learning rate
        self.adjust_learning_rate()

class TrainingState:
    def __init__(self):
        self.ngames = 0
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.max_score_count = 0  
        self.checkpoint_file = "pause.pth"
        self.training_log = []
        self.current_max_model = None  # Track current max score model file
    
    def save_max_score_model(self, agent, score):
        """Save the max score model and delete previous one"""
        new_model_file = f'max_score_{score}.safetensors'
        
        # Delete previous max score model if it exists
        if self.current_max_model and os.path.exists(self.current_max_model):
            try:
                os.remove(self.current_max_model)
                print(f"Deleted previous max score model: {self.current_max_model}")
            except Exception as e:
                print(f"Could not delete previous model: {e}")
        
        # Save new max score model
        agent.model.save(new_model_file)
        self.current_max_model = new_model_file
        print(f"New max score model saved: {new_model_file}")
    
    def save_checkpoint(self, agent):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
            'ngames': self.ngames,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
            'total_score': self.total_score,
            'record': self.record,
            'max_score_count': self.max_score_count,
            'training_log': self.training_log,
            'current_max_model': self.current_max_model,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.checkpoint_file)
        print(f"Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self, agent):
        """Load training checkpoint if exists"""
        if os.path.exists(self.checkpoint_file):
            try:
                checkpoint = torch.load(self.checkpoint_file)
                agent.model.load_state_dict(checkpoint['model_state_dict'])
                agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.ngames = checkpoint['ngames']
                self.plot_scores = checkpoint['plot_scores']
                self.plot_mean_scores = checkpoint['plot_mean_scores']
                self.total_score = checkpoint['total_score']
                self.record = checkpoint['record']
                self.max_score_count = checkpoint.get('max_score_count', 0)
                self.training_log = checkpoint.get('training_log', [])
                self.current_max_model = checkpoint.get('current_max_model', None)
                agent.ngames = self.ngames
                print(f"Resuming from checkpoint: Game {self.ngames}, Record: {self.record}")
                if self.current_max_model:
                    print(f" Current max model: {self.current_max_model}")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        return False

class AGENT:
    def __init__(self):
        self.ngames = 0
        self.epsilon = self.epsilon = max(5, 80 - self.ngames)
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.model = QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        # Get the position of the snake's head
        snake_head_x = game.snake.x[0]
        snake_head_y = game.snake.y[0]
        # Get the position of the food
        food_x = game.food.x  # Changed from apple to food
        food_y = game.food.y

        # Check for danger in each direction
        danger_straight = game.is_collision("straight")
        danger_right = game.is_collision("right") 
        danger_left = game.is_collision("left")

        # Get the current direction of the snake
        dir_up = game.snake.direction == "up"
        dir_down = game.snake.direction == "down"
        dir_left = game.snake.direction == "left"
        dir_right = game.snake.direction == "right"

        # Check where the food is relative to the snake's head
        food_up = food_y < snake_head_y     # Changed from apple to food
        food_down = food_y > snake_head_y
        food_left = food_x < snake_head_x
        food_right = food_x > snake_head_x

        # Build the state as a list of features
        state = [
            danger_straight,
            danger_right, 
            danger_left,
            dir_down,
            dir_left,
            dir_right,
            dir_up,
            food_down,      # Changed from apple to food
            food_left,
            food_up,
            food_right
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def train_on_long_memory(self):
        # Train on a batch of experiences from memory
        if len(self.memory) > BATCH:
            mini_sample = random.sample(self.memory, BATCH)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_on_short_memory(self, state, action, reward, next_state, done):
        # Train on the most recent experience
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decay epsilon for less exploration over time
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)  # Decay epsilon
        final_move = [0, 0, 0]
        
        # Exploration: choose a random move
        if random.uniform(0, 1) < self.epsilon / 100:  # Exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # Exploitation: choose the best move according to the model
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 # type: ignore
    
        return final_move

def plot(scores, mean_scores):
    # Plot the scores and mean scores over time
    plt.clf()
    plt.title("Training Progress - Snake AI")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores, label='Score', alpha=0.7)
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.axhline(y=MAX_POSSIBLE_SCORE, color='r', linestyle='--', label=f'Max Possible ({MAX_POSSIBLE_SCORE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.pause(0.1)

def train():
    # Initialize training state
    training_state = TrainingState()
    agent = AGENT()
    game = Game()
    plt.ion()
    
    # Try to load checkpoint
    if training_state.load_checkpoint(agent):
        print("Resuming training from checkpoint...")
    else:
        print("Starting new training session...")
    
    # Training parameters
    TARGET_SCORE = MAX_POSSIBLE_SCORE  # Ultimate goal
    MAX_GAMES = 10000  # Maximum training games
    SOLVED_THRESHOLD = min(50, MAX_POSSIBLE_SCORE * 0.8)  # 80% of max or 50, whichever is lower
    MAX_SCORE_THRESHOLD = 20  # Save final model after achieving max score this many times
    CHECKPOINT_INTERVAL = 100  # Save checkpoint every N games

    print(f" Training targets:")
    print(f"   Target Score: {TARGET_SCORE}")
    print(f"   Solved Threshold: {SOLVED_THRESHOLD}")
    print(f"   Max Games: {MAX_GAMES}")

    try:
        while training_state.ngames < MAX_GAMES:
            # Get the current state
            state_old = agent.get_state(game)
            # Decide on an action
            final_move = agent.get_action(state_old)
            # Play the game with the chosen action
            reward, done, score = game.play(final_move)
            # Get the new state after the action
            state_new = agent.get_state(game)
            
            # Train on the most recent experience
            agent.train_on_short_memory(state_old, final_move, reward, state_new, done)
            # Store the experience in memory
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Reset the game if done
                game.reset()
                training_state.ngames += 1
                agent.ngames = training_state.ngames
                # Train on a batch of experiences from memory
                agent.train_on_long_memory()
                
                # Update scores and save max score model
                if score > training_state.record:
                    training_state.record = score
                    training_state.save_max_score_model(agent, score)

                training_state.plot_scores.append(score)
                training_state.total_score += score
                mean_score = training_state.total_score / training_state.ngames
                training_state.plot_mean_scores.append(mean_score)
                
                # Check if max possible score achieved
                if score >= TARGET_SCORE:
                    training_state.max_score_count += 1
                    print(f"MAX SCORE ACHIEVED! ({training_state.max_score_count}/{MAX_SCORE_THRESHOLD})")
                
                # Log training progress
                log_entry = {
                    'game': training_state.ngames,
                    'score': score,
                    'mean_score': mean_score,
                    'record': training_state.record,
                    'epsilon': agent.epsilon,
                    'timestamp': datetime.now().isoformat()
                }
                training_state.training_log.append(log_entry)
                
                print(f'Game {training_state.ngames}, Score {score}, Record: {training_state.record}, Mean: {mean_score:.1f}, ε: {agent.epsilon:.1f}')
                plot(training_state.plot_scores, training_state.plot_mean_scores)
                
                # Check stopping conditions
                if training_state.max_score_count >= MAX_SCORE_THRESHOLD:
                    print(f" MASTERY ACHIEVED! Max score reached {training_state.max_score_count} times!")
                    agent.model.save('final_master_model.safetensors')
                    break
                    
                if mean_score >= SOLVED_THRESHOLD and training_state.ngames >= 100:
                    print(f" SOLVED! Mean score: {mean_score:.1f} over {training_state.ngames} games")
                    agent.model.save('final_solved_model.safetensors')
                    break
                
                # Save checkpoint periodically
                if training_state.ngames % CHECKPOINT_INTERVAL == 0:
                    training_state.save_checkpoint(agent)

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Training stopped by user")
                    training_state.save_checkpoint(agent)
                    pygame.quit()
                    return

    except KeyboardInterrupt:
        # Save checkpoint if training interrupted by user
        training_state.save_checkpoint(agent)
    except Exception as e:
        # Save checkpoint if any error occurs
        training_state.save_checkpoint(agent)
        raise
    
    # Training completed
    print(f"\n Training completed after {training_state.ngames} games!")
    print(f" Final Record: {training_state.record}")
    print(f" Final Mean Score: {training_state.total_score/training_state.ngames:.1f}")
    print(f"Max Score Achieved: {training_state.max_score_count} times")
    
    # Save final model and training log
    agent.model.save(f'final_model_games_{training_state.ngames}.safetensors')
    
    # Save training log
    with open(f'training_log_{training_state.ngames}.json', 'w') as f:
        json.dump(training_state.training_log, f, indent=2)
    
    # Clean up checkpoint file
    if os.path.exists(training_state.checkpoint_file):
        os.remove(training_state.checkpoint_file)
        print("Checkpoint file cleaned up")
    
    # Keep plot open
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()