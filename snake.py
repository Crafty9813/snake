"""Snake game using tkinter (no external dependencies).

Controls: arrow keys or WASD to move, P to pause, R/Space to restart, Esc to quit.
"""

import random
import tkinter as tk

CELL = 20
COLS = 30
ROWS = 24
WIDTH = COLS * CELL
HEIGHT = ROWS * CELL
START_DELAY = 120  # ms per tick
MIN_DELAY = 50
SPEEDUP = 3  # ms faster per food eaten

BG = "#1e1e2e"
SNAKE_HEAD = "#a6e3a1"
SNAKE_BODY = "#40a02b"
FOOD = "#f38ba8"
TEXT = "#cdd6f4"

DIRECTIONS = {
    "Up": (0, -1), "w": (0, -1),
    "Down": (0, 1), "s": (0, 1),
    "Left": (-1, 0), "a": (-1, 0),
    "Right": (1, 0), "d": (1, 0),
}


class SnakeGame:
    def __init__(self, root):
        self.root = root
        root.title("Snake")
        root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=BG, highlightthickness=0)
        self.canvas.pack()
        self.high_score = 0
        self.after_id = None
        root.bind("<Key>", self.on_key)
        self.reset()

    def reset(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)
        mid_x, mid_y = COLS // 2, ROWS // 2
        self.snake = [(mid_x - i, mid_y) for i in range(3)]
        self.direction = (1, 0)
        self.pending = []  # queued direction changes, so quick key presses aren't lost
        self.score = 0
        self.delay = START_DELAY
        self.paused = False
        self.game_over = False
        self.place_food()
        self.tick()

    def place_food(self):
        free = {(x, y) for x in range(COLS) for y in range(ROWS)} - set(self.snake)
        self.food = random.choice(list(free)) if free else None

    def on_key(self, event):
        key = event.keysym
        if key == "Escape":
            self.root.destroy()
        elif key in ("r", "R", "space") and self.game_over:
            self.reset()
        elif key in ("p", "P") and not self.game_over:
            self.paused = not self.paused
            self.draw()
        elif key.lower() in DIRECTIONS or key in DIRECTIONS:
            new = DIRECTIONS.get(key) or DIRECTIONS[key.lower()]
            last = self.pending[-1] if self.pending else self.direction
            # Ignore reversing into yourself or repeating the same direction
            if new != last and (new[0] + last[0], new[1] + last[1]) != (0, 0) and len(self.pending) < 3:
                self.pending.append(new)

    def tick(self):
        if not self.paused and not self.game_over:
            self.step()
        self.draw()
        self.after_id = self.root.after(self.delay, self.tick)

    def step(self):
        if self.pending:
            self.direction = self.pending.pop(0)
        hx, hy = self.snake[0]
        dx, dy = self.direction
        head = (hx + dx, hy + dy)

        ate = head == self.food
        # The tail moves out of the way this tick unless we're growing
        body = self.snake if ate else self.snake[:-1]
        if not (0 <= head[0] < COLS and 0 <= head[1] < ROWS) or head in body:
            self.game_over = True
            self.high_score = max(self.high_score, self.score)
            return

        self.snake.insert(0, head)
        if ate:
            self.score += 1
            self.delay = max(MIN_DELAY, self.delay - SPEEDUP)
            self.place_food()
            if self.food is None:  # board filled: you win
                self.game_over = True
                self.high_score = max(self.high_score, self.score)
        else:
            self.snake.pop()

    def draw(self):
        c = self.canvas
        c.delete("all")
        if self.food:
            fx, fy = self.food
            c.create_oval(fx * CELL + 3, fy * CELL + 3, (fx + 1) * CELL - 3, (fy + 1) * CELL - 3,
                          fill=FOOD, outline="")
        for i, (x, y) in enumerate(self.snake):
            c.create_rectangle(x * CELL + 1, y * CELL + 1, (x + 1) * CELL - 1, (y + 1) * CELL - 1,
                               fill=SNAKE_HEAD if i == 0 else SNAKE_BODY, outline="")
        c.create_text(8, 6, anchor="nw", fill=TEXT, font=("Helvetica", 12),
                      text=f"Score: {self.score}   Best: {self.high_score}")
        if self.game_over:
            title = "You win!" if self.food is None else "Game Over"
            self.overlay(title, f"Score: {self.score}\nPress R or Space to play again")
        elif self.paused:
            self.overlay("Paused", "Press P to resume")

    def overlay(self, title, subtitle):
        self.canvas.create_text(WIDTH / 2, HEIGHT / 2 - 20, fill=TEXT,
                                font=("Helvetica", 32, "bold"), text=title)
        self.canvas.create_text(WIDTH / 2, HEIGHT / 2 + 25, fill=TEXT,
                                font=("Helvetica", 14), text=subtitle, justify="center")


if __name__ == "__main__":
    root = tk.Tk()
    SnakeGame(root)
    root.mainloop()
