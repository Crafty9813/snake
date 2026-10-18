"""Train a DQN agent to play snake.

Usage:
    python train.py                     # default 1000 episodes
    python train.py --episodes 3000 --out models/snake_dqn.pt
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from agent import DQNAgent
from snake_env import N_ACTIONS, STATE_SIZE, SnakeEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--cols", type=int, default=30)
    p.add_argument("--rows", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--eps-decay-episodes", type=int, default=300,
                   help="episodes over which epsilon decays linearly to eps-end")
    p.add_argument("--train-every", type=int, default=4, help="env steps per gradient update")
    p.add_argument("--out", default="models/snake_dqn.pt", help="where to save the best model")
    p.add_argument("--log", default="models/training_log.csv")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    env = SnakeEnv(args.cols, args.rows, seed=args.seed)
    agent = DQNAgent(STATE_SIZE, N_ACTIONS, hidden=args.hidden, lr=args.lr, gamma=args.gamma,
                     batch_size=args.batch_size, device=args.device)

    print(f"Training on {args.device} for {args.episodes} episodes "
          f"({args.cols}x{args.rows} board). Best model -> {args.out}")

    scores = []
    best_avg = -1.0
    total_steps = 0
    start = time.time()

    with open(args.log, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["episode", "score", "avg100", "epsilon", "steps"])

        for episode in range(1, args.episodes + 1):
            frac = min(1.0, (episode - 1) / max(1, args.eps_decay_episodes))
            epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_steps += 1
                if total_steps % args.train_every == 0:
                    agent.learn()

            scores.append(info["score"])
            avg100 = float(np.mean(scores[-100:]))
            log.writerow([episode, info["score"], f"{avg100:.2f}", f"{epsilon:.3f}", env.steps])

            # Only save once epsilon has mostly decayed, so the average reflects the policy
            if episode >= min(100, args.episodes) and avg100 > best_avg:
                best_avg = avg100
                agent.save(args.out, episode=episode, avg100=avg100, cols=args.cols, rows=args.rows)

            if episode % 50 == 0 or episode == args.episodes:
                print(f"ep {episode:5d} | score {info['score']:3d} | avg100 {avg100:6.2f} | "
                      f"best {max(scores):3d} | eps {epsilon:.3f} | {time.time() - start:6.1f}s")

    final_path = os.path.splitext(args.out)[0] + "_final.pt"
    agent.save(final_path, episode=args.episodes, avg100=float(np.mean(scores[-100:])),
               cols=args.cols, rows=args.rows)
    print(f"Done. Best avg100 {best_avg:.2f} saved to {args.out}; final model saved to {final_path}")


if __name__ == "__main__":
    main()
