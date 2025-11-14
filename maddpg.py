import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ============================================================
# Réseaux Actor / Critic (tu peux garder les tiens si identiques)
# ============================================================

class Actor(nn.Module):
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
        return self.net(x)


class Critic(nn.Module):
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
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# ============================================================
# Replay buffer (un échantillon = un prédateur)
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


class MADDPGAgent:
    """
    Version 'partagée' : un seul actor/critic pour tous les prédateurs,
    inspirée de la partie 'predator' du code du papier.
    L’API reste compatible avec ton train.py :
      - select_action(state, agent_idx, noise_scale)
      - store_transition(obs, actions, rewards, next_obs)
      - update()
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents=3,          # conservé pour compat, mais non utilisé
        gamma=0.95,
        tau=0.01,
        lr=1e-3,
        batch_size=128,
        device="cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Réseaux principal + cibles (partagés par tous les prédateurs)
        self.actor = Actor(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=128).to(self.device)

        self.actor_target = Actor(state_dim, action_dim, hidden_dim=128).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=128).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(capacity=100000)

        # stats de normalisation (option simple : recomputées à chaque update)
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)

    # --------------------------------------------------------
    def select_action(self, state, agent_idx=None, noise_scale=0.1):
        """
        state : np.array (STATE_DIM,) pour UN prédateur.
        agent_idx est ignoré (compat train.py).
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
        states, next_states : shape (n_agents, state_dim)
        actions : shape (n_agents, action_dim)
        rewards : shape (n_agents,)
        On découpe et on ajoute 1 expérience par prédateur.
        """
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)

        n = states.shape[0]
        for i in range(n):
            self.buffer.push(
                states[i],
                actions[i],
                rewards[i],
                next_states[i]
            )

    # --------------------------------------------------------
    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        # normalisation (simple) sur ce batch
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0) + 1e-8

        norm_states = (states - self.state_mean) / self.state_std
        norm_next_states = (next_states - self.state_mean) / self.state_std

        # ---------- Critic ----------
        with torch.no_grad():
            next_actions = self.actor_target(norm_next_states)
            target_q = rewards + self.gamma * self.critic_target(
                norm_next_states, next_actions
            )

        current_q = self.critic(norm_states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------- Actor ----------
        current_actions = self.actor(norm_states)
        actor_loss = -self.critic(norm_states, current_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ---------- Soft update cibles ----------
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
