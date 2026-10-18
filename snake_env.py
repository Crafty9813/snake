"""Headless snake env for RL.

Same rules as snake.py (walls and self-collision kill, food grows the snake)

Actions relative to current heading: 0 = straight, 1 = turn right, 2 = turn left.
"""

import random

import numpy as np

# Clockwise order (screen coordinates, y grows downwards): right, down, left, up
CLOCKWISE = [(1, 0), (0, 1), (-1, 0), (0, -1)]

STRAIGHT, RIGHT, LEFT = 0, 1, 2
N_ACTIONS = 3
STATE_SIZE = 11

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0


class SnakeEnv:
    def __init__(self, cols=30, rows=24, seed=None):
        self.cols = cols
        self.rows = rows
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        mid_x, mid_y = self.cols // 2, self.rows // 2
        self.snake = [(mid_x - i, mid_y) for i in range(3)]
        self.direction = (1, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.place_food()
        return self.get_state()

    def place_food(self):
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.cols) for y in range(self.rows) if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def turn(self, action):
        idx = CLOCKWISE.index(self.direction)
        if action == RIGHT:
            idx = (idx + 1) % 4
        elif action == LEFT:
            idx = (idx - 1) % 4
        return CLOCKWISE[idx]

    def is_collision(self, point, body=None):
        x, y = point
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return True
        # By default ignore the tail: it moves out of the way on the next step
        body = self.snake[:-1] if body is None else body
        return point in body

    def step(self, action):
        """Advance one tick. Returns (state, reward, done, info)."""
        self.steps += 1
        self.steps_since_food += 1
        self.direction = self.turn(action)
        hx, hy = self.snake[0]
        dx, dy = self.direction
        head = (hx + dx, hy + dy)

        ate = head == self.food
        body = self.snake if ate else self.snake[:-1]
        # Also end the game if the snake loops forever without eating
        starved = self.steps_since_food > 100 * len(self.snake)
        if self.is_collision(head, body) or starved:
            return self.get_state(), REWARD_DEATH, True, {"score": self.score}

        self.snake.insert(0, head)
        if ate:
            self.score += 1
            self.steps_since_food = 0
            self.place_food()
            done = self.food is None  # board filled
            return self.get_state(), REWARD_FOOD, done, {"score": self.score}

        self.snake.pop()
        return self.get_state(), 0.0, False, {"score": self.score}

    def get_state(self):
        """11 binary features describing the snake's surroundings

        [danger straight, danger right, danger left,
         moving left, moving right, moving up, moving down,
         food left, food right, food up, food down]
        """
        hx, hy = self.snake[0]

        def danger(action):
            dx, dy = self.turn(action)
            return self.is_collision((hx + dx, hy + dy))

        dx, dy = self.direction
        fx, fy = self.food if self.food else (hx, hy)
        return np.array([
            danger(STRAIGHT),
            danger(RIGHT),
            danger(LEFT),
            (dx, dy) == (-1, 0),
            (dx, dy) == (1, 0),
            (dx, dy) == (0, -1),
            (dx, dy) == (0, 1),
            fx < hx,
            fx > hx,
            fy < hy,
            fy > hy,
        ], dtype=np.float32)
