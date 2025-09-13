# Snake AI with Deep Q-Learning

This project implements a Deep Q-Learning (DQN) agent to play the classic Snake game using [PyTorch] and [Pygame]. The agent learns to play Snake through reinforcement learning, optimizing its score over thousands of games.

## Features

- **Deep Q-Network**: Uses a neural network to approximate Q-values for state-action pairs.
- **Experience Replay**: Stores past experiences to train on random batches for stability.
- **Checkpointing**: Automatically saves and resumes training progress.
- **Training Visualization**: Plots score and mean score during training.
- **Model Saving**: Saves the best and final models using the [safetensors] format.

## Project Structure

- [`agent.py`](agent.py): Main training loop, DQN agent, neural network, and training utilities.
- [`snake_game.py`](snake_game.py): Pygame implementation of the Snake game environment.
- [`requirements.txt`](requirements.txt): Python dependencies.
- `resources/`: Contains images for the game (e.g., `apple.jpg`, `block.jpg`).
- `training_log_*.json`: Training logs saved after training sessions.
- `myvenv/`: (Optional) Python virtual environment.

## Installation

1. **Clone the repository** and navigate to the project directory.

2. **Create a virtual environment** (optional but recommended):

    ```sh
    python -m venv myvenv
    source myvenv/Scripts/activate  # On Windows
    # or
    source myvenv/bin/activate      # On Linux/Mac
    ```

3. **Install dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

To start training the agent, run:

```sh
python agent.py
```

- Training progress will be displayed in the console and as a live plot.
- Checkpoints are saved periodically and loaded automatically if present.
- Final models and training logs are saved after training.

## Customization

- **Game parameters** (grid size, block size, etc.) can be adjusted in [`snake_game.py`](snake_game.py).
- **Hyperparameters** (learning rate, batch size, epsilon decay, etc.) can be tuned in [`agent.py`](agent.py).

## Requirements

- Python 3.8+
- See [`requirements.txt`](requirements.txt) for all dependencies.

---
