import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ============================================================
# 🔹 Réseaux Actor et Critic
# ============================================================

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # actions normalisées entre -1 et 1
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1))


# ============================================================
# 🔹 Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 🔹 MADDPG Agent
# ============================================================

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, n_agents=3, gamma=0.95, tau=0.01, lr=1e-3, device="cpu"):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(n_agents)]
        self.critics = [Critic(state_dim * n_agents, action_dim * n_agents).to(device) for _ in range(n_agents)]

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=lr) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=lr) for c in self.critics]

        self.buffer = ReplayBuffer()
        self.batch_size = 128

    def select_action(self, state, agent_idx, noise_scale=0.1):
        """Renvoie une action bruitée (exploration)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actors[agent_idx](state).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

    def update(self):
        """Met à jour les acteurs et critiques"""
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        for i in range(self.n_agents):
            # --- Critic update ---
            with torch.no_grad():
                next_actions = [self.actors[j](next_states[:, j, :]) for j in range(self.n_agents)]
                next_actions = torch.cat(next_actions, dim=1)
                target_q = rewards[:, i].unsqueeze(1) + self.gamma * self.critics[i](next_states.view(self.batch_size, -1), next_actions)

            current_q = self.critics[i](states.view(self.batch_size, -1), actions.view(self.batch_size, -1))
            critic_loss = nn.MSELoss()(current_q, target_q)

            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # --- Actor update ---
            actor_actions = [self.actors[j](states[:, j, :]) if j == i else actions[:, j, :] for j in range(self.n_agents)]
            actor_actions = torch.cat(actor_actions, dim=1)
            actor_loss = -self.critics[i](states.view(self.batch_size, -1), actor_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

    def store_transition(self, state, action, reward, next_state):
        self.buffer.push(state, action, reward, next_state)
