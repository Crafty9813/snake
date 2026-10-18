"""Watch a trained agent play snake in a tkinter window.

Usage:
    python play.py                          # loads models/snake_dqn.pt
    python play.py --model models/snake_dqn_final.pt --speed 30

Keys: +/- change speed, P pause, Esc quit.
"""

import argparse
import tkinter as tk

from agent import DQNAgent
from snake import BG, CELL, FOOD, SNAKE_BODY, SNAKE_HEAD, TEXT
from snake_env import SnakeEnv


class Viewer:
    def __init__(self, root, agent, env, delay):
        self.root = root
        self.agent = agent
        self.env = env
        self.delay = delay
        self.paused = False
        self.games = 0
        self.best = 0
        self.last_score = 0

        root.title("Snake AI")
        root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=env.cols * CELL, height=env.rows * CELL,
                                bg=BG, highlightthickness=0)
        self.canvas.pack()
        root.bind("<Key>", self.on_key)

        self.state = env.reset()
        self.tick()

    def on_key(self, event):
        key = event.keysym
        if key == "Escape":
            self.root.destroy()
        elif key in ("p", "P"):
            self.paused = not self.paused
        elif key in ("plus", "equal", "KP_Add"):
            self.delay = max(1, self.delay // 2)
        elif key in ("minus", "KP_Subtract"):
            self.delay = min(1000, self.delay * 2)

    def tick(self):
        if not self.paused:
            action = self.agent.act(self.state)
            self.state, _, done, info = self.env.step(action)
            if done:
                self.games += 1
                self.last_score = info["score"]
                self.best = max(self.best, info["score"])
                self.state = self.env.reset()
        self.draw()
        self.root.after(self.delay, self.tick)

    def draw(self):
        c = self.canvas
        c.delete("all")
        if self.env.food:
            fx, fy = self.env.food
            c.create_oval(fx * CELL + 3, fy * CELL + 3, (fx + 1) * CELL - 3, (fy + 1) * CELL - 3,
                          fill=FOOD, outline="")
        for i, (x, y) in enumerate(self.env.snake):
            c.create_rectangle(x * CELL + 1, y * CELL + 1, (x + 1) * CELL - 1, (y + 1) * CELL - 1,
                               fill=SNAKE_HEAD if i == 0 else SNAKE_BODY, outline="")
        c.create_text(8, 6, anchor="nw", fill=TEXT, font=("Helvetica", 12),
                      text=f"Score: {self.env.score}   Last: {self.last_score}   "
                           f"Best: {self.best}   Games: {self.games}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/snake_dqn.pt")
    p.add_argument("--speed", type=int, default=60, help="ms per move")
    args = p.parse_args()

    agent, ckpt = DQNAgent.load(args.model)
    env = SnakeEnv(ckpt.get("cols", 30), ckpt.get("rows", 24))
    print(f"Loaded {args.model} (episode {ckpt.get('episode')}, avg100 {ckpt.get('avg100', 0):.2f})")

    root = tk.Tk()
    Viewer(root, agent, env, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
