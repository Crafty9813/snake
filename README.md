# snake24

Snake in Python (tkinter), plus a PyTorch DQN agent that learns to play it.

## Play

```bash
python3 snake.py
```

Arrow keys / WASD to move, P to pause, R or Space to restart, Esc to quit.

## Train an AI

```bash
python3 -m venv .venv
# CPU only:
.venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# NVIDIA GPU (needs a working driver, check with nvidia-smi):
.venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple

.venv/bin/python train.py --episodes 1000      # uses CUDA automatically if available
.venv/bin/python play.py                        # watch the best model play
```

- `snake_env.py`: headless game with the same rules as `snake.py`
- `agent.py`: Double DQN (MLP Q-network, target network, replay buffer)
- `train.py`: training loop; saves the best model (by 100-game average) to `models/snake_dqn.pt` and logs to `models/training_log.csv`
- `play.py`: renders the trained agent in a window (`+`/`-` change speed)

Run `python train.py --help` for hyperparameters.
