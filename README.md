# Snake AI: Deep Q-Learning Agent

A reinforcement learning project demonstrating a Deep Q-Network (DQN) agent that learns to play the classic Snake game through self-play and optimization. Built with **PyTorch**, **Pygame**, and **NumPy**.

## Overview

This project implements sophisticated deep reinforcement learning techniques to train an autonomous agent that learns optimal gameplay strategies from scratch. The agent uses a neural network to approximate Q-values and employs experience replay for stable learning, achieving measurable improvements in game performance over thousands of training episodes.

**Key Learning Objectives Demonstrated:**
- Deep Reinforcement Learning (DQN Architecture)
- Experience Replay & Epsilon-Greedy Exploration-Exploitation
- Adaptive Learning Rate Scheduling
- State Representation & Policy Learning
- Model Checkpointing & Resume Functionality

## Technical Architecture

### Neural Network (QNet)
- **Input Layer**: 11-dimensional state vector (danger detection, direction, food position)
- **Hidden Layers**: 2 fully-connected layers (256 neurons each) with ReLU activation
- **Output Layer**: 3 actions (straight, left turn, right turn)
- **Model Format**: SafeTensors (efficient and secure model serialization)

### Training Strategy
- **Algorithm**: Deep Q-Learning with Target Network updates
- **Memory**: Prioritized experience replay buffer (100K capacity)
- **Discount Factor (γ)**: 0.98 (emphasis on long-term rewards)
- **Learning Rate**: Adaptive decay from 0.001 to 0.00001
- **Exploration**: ε-greedy strategy with logarithmic decay (100 → 1)
- **Reward Shaping**: 
  - Eating food: +10 (+ snake length bonus)
  - Getting closer to food: +0.1
  - Step penalty: -0.01 (encourages efficiency)
  - Collision: -10

### Game Environment
- **Grid Size**: 11×11 (121 total squares, max theoretical score: 120)
- **State Space**: Binary features for obstacle detection, direction, and relative food position
- **Action Space**: 3 discrete actions with collision prevention

## Features

**Advanced Capabilities:**
- **Smart Training**: Automatically resumes from checkpoints, enabling long training sessions
- **Adaptive Learning**: Dynamic learning rate scheduling for better convergence
- **Progress Tracking**: Real-time visualization of training metrics (score, mean score vs. theoretical maximum)
- **Model Persistence**: 
  - Saves best model at each new record
  - Saves final models upon training completion
  - Automatic checkpoint management
- **Training Telemetry**: JSON logging with detailed metrics per episode
- **Robust Error Handling**: Graceful checkpointing on interruption

## Project Structure

```
snake-reinforcement-learning/
├── agent.py              # DQN agent, neural network, training loop
├── snake_game.py         # Game environment (Pygame)
├── requirements.txt      # Python dependencies
├── resources/            # Game assets
├── *.safetensors         # Trained model files (best & final)
├── training_log_*.json   # Training statistics & telemetry
└── README.md
```

| File | Purpose |
|------|---------|
| [agent.py](agent.py) | QNet (neural network), QTrainer (backprop), AGENT (policy), TrainingState (checkpoint mgmt) |
| [snake_game.py](snake_game.py) | Game simulation: Snake class, Food class, collision detection, reward calculation |
| [requirements.txt](requirements.txt) | Dependencies: torch, pygame, safetensors, matplotlib |

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Clone & Navigate
```sh
git clone <repository-url>
cd snake-reinforcement-learning
```

### Step 2: Create Virtual Environment (Recommended)
```sh
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies
```sh
pip install -r requirements.txt
```

## Training & Evaluation

### Start Training
```sh
python agent.py
```

**What happens during training:**
1. Agent initializes with random policy (100% exploration)
2. Each episode: agent observes state → selects action → receives feedback
3. Experience replay: agent learns from random batches of past experiences
4. Epsilon decay: gradually shifts from exploration to exploitation
5. Real-time plot shows training progress
6. Checkpoints saved every 100 games; best models saved automatically

### Training Milestones

The training process targets three objectives:

1. **Solved Threshold** (~80% of max score): Early convergence indicator
2. **Target Score** (120 - theoretical maximum): Ultimate goal
3. **Mastery** (20 consecutive max scores): Demonstrates consistent optimal play

### Output Files
- `final_master_model.safetensors` - Model after achieving mastery
- `final_solved_model.safetensors` - Model reaching solved threshold
- `final_model_games_*.safetensors` - Model at training completion
- `training_log_*.json` - Complete training telemetry
- `max_score_*.safetensors` - Best model at each new record

## Results & Performance

Training metrics are logged in real-time:
- Per-episode score and mean score
- Record (best) score tracking
- Epsilon (exploration rate) decay
- Model checkpoints for reproducibility

Example output:
```
Game 1000, Score 15, Record: 47, Mean: 8.2, ε: 3.5
Game 2000, Score 31, Record: 52, Mean: 12.1, ε: 1.2
Game 3000, Score 52, Record: 52, Mean: 18.5, ε: 0.8
```

## Technical Highlights for Evaluation

🎓 **Demonstrates Proficiency In:**
- **Reinforcement Learning**: DQN fundamentals, value-based learning, temporal difference
- **Deep Learning**: PyTorch, neural network design, loss functions, backpropagation
- **Python**: Object-oriented design, error handling, JSON/serialization, datetime handling
- **ML Engineering**: Model persistence, checkpointing, hyperparameter tuning, logging
- **Game Development**: State representation, collision detection, reward shaping
- **Data Visualization**: Matplotlib for training progress tracking

## Customization

Key hyperparameters in [agent.py](agent.py#L18-L28):

```python
MAX_MEM = 100_000       # Experience replay buffer size
BATCH = 1000            # Training batch size
LR = 0.001              # Initial learning rate
GAMMA = 0.98            # Discount factor
EPSILON_DECAY = 0.99    # Exploration decay rate
```

Adjusting these parameters allows experimentation with different learning dynamics.

## Future Enhancements

- Dueling DQN architecture (separate value & advantage streams)
- Prioritized Experience Replay (PER)
- Double DQN for reduced overestimation
- Vision-based input (CNN) instead of hand-crafted features
- Multi-agent competitive scenarios

## Requirements

See [requirements.txt](requirements.txt):
- **torch** - Deep learning framework
- **pygame** - Game rendering
- **numpy** - Numerical computation
- **matplotlib** - Visualization
- **safetensors** - Secure model serialization

## Customization

- **Game parameters** (grid size, block size, etc.) can be adjusted in [`snake_game.py`](snake_game.py).
- **Hyperparameters** (learning rate, batch size, epsilon decay, etc.) can be tuned in [`agent.py`](agent.py).

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies.

---
