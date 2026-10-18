"""Double DQN agent: Q-network, target network, and experience replay."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_size, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, n_actions, hidden=256, lr=1e-3, gamma=0.9,
                 buffer_size=100_000, batch_size=512, target_sync=1000, device="cpu"):
        self.state_size = state_size
        self.n_actions = n_actions
        self.hidden = hidden
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.device = torch.device(device)

        self.q = QNet(state_size, n_actions, hidden).to(self.device)
        self.target = QNet(state_size, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            return int(self.q(s).argmax(dim=1).item())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = (
            t.to(self.device) for t in self.memory.sample(self.batch_size)
        )

        q_values = self.q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net scores it
            next_actions = self.q(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()

    def save(self, path, **extra):
        torch.save({
            "state_dict": self.q.state_dict(),
            "state_size": self.state_size,
            "n_actions": self.n_actions,
            "hidden": self.hidden,
            **extra,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        agent = cls(ckpt["state_size"], ckpt["n_actions"], hidden=ckpt["hidden"], device=device)
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.target.load_state_dict(ckpt["state_dict"])
        agent.q.eval()
        return agent, ckpt
