import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from networks import Actor, Critic  # ou réutilise tes Actor/Critic actuels


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state, action, etc. sont UN agent; on empile tous les agents un par un
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=0.1):
        self.action_dim = action_dim
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.action_dim)


class MADDPGPredators:
    """
    Un seul acteur/critic partagé pour tous les prédateurs,
    inspiré de la partie 'predator' de leur MADDPG.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.01,
        buffer_size=100000,
        batch_size=256,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # réseau principal
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        # cibles
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim + action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.noise = GaussianNoise(action_dim)

        # normalisation des états (facultatif mais proche du repo)
        self.state_mean = torch.zeros(state_dim).to(self.device)
        self.state_std = torch.ones(state_dim).to(self.device)

    # ---------------------------------------------------------
    def select_actions(self, states, add_noise=True):
        """
        states : np.array shape (n_predators, state_dim)
        retourne actions : np.array shape (n_predators, action_dim)
        """
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        # normalisation
        norm_states = (states_t - self.state_mean) / (self.state_std + 1e-8)

        with torch.no_grad():
            acts = self.actor(norm_states).cpu().numpy()

        if add_noise:
            acts += np.stack([self.noise.sample() for _ in range(len(states))], axis=0)
            acts = np.clip(acts, -1.0, 1.0)

        return acts

    # ---------------------------------------------------------
    def store_transition(self, states, actions, rewards, next_states, dones):
        """
        states, next_states : (n_predators, state_dim)
        actions : (n_predators, action_dim)
        rewards : (n_predators,)
        dones   : (n_predators,)
        On stocke un tuple par prédateur, comme dans leur ReplayBuffer.
        """
        n = states.shape[0]
        for i in range(n):
            self.buffer.push(
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                dones[i]
            )

    # ---------------------------------------------------------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # maj stats de normalisation
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0)

        norm_states = (states - self.state_mean) / (self.state_std + 1e-8)
        norm_next_states = (next_states - self.state_mean) / (self.state_std + 1e-8)

        # -------- Critic update --------
        with torch.no_grad():
            next_actions = self.actor_target(norm_next_states)
            target_q = rewards + self.gamma * self.critic_target(
                torch.cat([norm_next_states, next_actions], dim=1)
            ) * (1 - dones)

        current_q = self.critic(torch.cat([norm_states, actions], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # -------- Actor update --------
        current_actions = self.actor(norm_states)
        actor_loss = -self.critic(torch.cat([norm_states, current_actions], dim=1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # -------- Soft update cibles --------
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

