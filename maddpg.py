"""
maddpg.py

Minimal MADDPG-style implementation used for collective behavior experiments.

This module provides:
- Actor and Critic neural networks (shared across agents).
- A simple replay buffer storing per-agent transitions.
- A MADDPGAgent class with:
    - Shared actor / critic for all agents
    - Target networks with soft updates
    - Action selection with optional exploration noise
    - Batch normalization of states (running mean/std updated per batch)

Note:
- Although named MADDPG, this implementation uses a shared actor and critic
  (homogeneous agents), not fully centralized critics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ------------------------------------------------------------
# Actor / Critic networks

class Actor(nn.Module):
    """
    Actor network mapping a state vector to a continuous action in [-1, 1]^action_dim.
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass of the actor network."""
        return self.net(x)


class Critic(nn.Module):
    """
    Critic network estimating Q(s, a) for a single agent.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """Forward pass of the critic network."""
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# ------------------------------------------------------------
# Replay buffer

class ReplayBuffer:
    """
    Simple FIFO replay buffer storing individual agent transitions.
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        """
        Store a transition tuple in the buffer.
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

# ------------------------------------------------------------
# MADDPG agent (shared actor for all agents)

class MADDPGAgent:
    """
    MADDPG-style agent with a shared actor and critic for homogeneous agents.

    Features:
    - Shared policy across all agents
    - Per-agent transitions stored independently
    - Soft target updates
    - State normalization based on batch statistics
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents=3,
        gamma=0.95,
        tau=0.01,
        lr=1e-3,
        batch_size=128,
        device="cpu"
    ):
        """Initialize networks, optimizers, replay buffer, and hyperparameters."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Main networks
        self.actor = Actor(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=128).to(self.device)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=128).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=100000)

        # State normalization statistics
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)

    # --------------------------------------------------------
    def select_action(self, state, agent_idx=None, noise_scale=0.1):
        """
        Select an action for a single agent state.

        Args:
            state: np.array of shape (state_dim,)
            agent_idx: unused (kept for backward compatibility)
            noise_scale: standard deviation of Gaussian exploration noise

        Returns:
            Action clipped to [-1, 1]^action_dim
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_state = (state_t - self.state_mean) / (self.state_std + 1e-8)

        with torch.no_grad():
            action = self.actor(norm_state).cpu().numpy()[0]

        if noise_scale > 0.0:
            action += noise_scale * np.random.randn(*action.shape)

        return np.clip(action, -1.0, 1.0)

    # --------------------------------------------------------
    def store_transition(self, states, actions, rewards, next_states):
        """
        Store transitions for all agents (one transition per agent).

        Args:
            states, next_states: (n_agents, state_dim)
            actions: (n_agents, action_dim)
            rewards: (n_agents,)
        """
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)

        n = states.shape[0]
        for i in range(n):
            self.buffer.push(states[i], actions[i], rewards[i], next_states[i])

    # --------------------------------------------------------
    def update(self):
        """
        Perform one gradient update step for actor and critic.
        """
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # Update normalization statistics from current batch
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0) + 1e-8

        norm_states = (states - self.state_mean) / self.state_std
        norm_next_states = (next_states - self.state_mean) / self.state_std

        # ----- Critic update -----
        with torch.no_grad():
            next_actions = self.actor_target(norm_next_states)
            target_q = rewards + self.gamma * self.critic_target(
                norm_next_states, next_actions
            )

        current_q = self.critic(norm_states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----- Actor update -----
        current_actions = self.actor(norm_states)
        actor_loss = -self.critic(norm_states, current_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ----- Soft target updates -----
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        """
        Soft update of target network parameters.
        """
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
