import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_relu_seq = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_features)
        )

    def forward(self, x):
        return self.fc_relu_seq(x)


class Agent:
    def __init__(self, net, env, gamma=0.99, optimizer=optim.Adam, lr=1e-3):
        self.net = net
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer(net.parameters(), lr=lr)
        self.buffer = []

    def _select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        # print('STATE SHAPE: ', state.shape)
        with torch.no_grad():
            logits = self.net(state)
            probs = F.softmax(logits, dim=-1)

        action = torch.multinomial(probs, num_samples=1).item()
        return action, logits

    def _gamma_rewards(self, rewards):
        res = []
        sum_r = 0.0
        for r in reversed(rewards):
            sum_r = sum_r * self.gamma + r
            res.append(sum_r)
        return list(reversed(res))

    def _prepare_buffer(self, gamma_rewards):  # ToDo rewrite logic
        assert len(self.buffer) == len(gamma_rewards)
        self.buffer = torch.Tensor(self.buffer)
        gamma_rewards = torch.Tensor(gamma_rewards).reshape(-1, 1)
        gamma_rewards = (gamma_rewards -gamma_rewards.mean()) / (gamma_rewards.std() + 1e-8) # Normalizing
        self.buffer = torch.hstack([self.buffer, gamma_rewards]) # s, a, r, s', R
        self.states = self.buffer[:, 0]
        self.actions = self.buffer[:, 1]

    def _extract_rewards(self):
        rewards = [row[2] for row in self.buffer]
        return rewards

    def _play_episode(self):
        self.buffer = []
        obs, info = self.env.reset()
        episode_logits = []
        while True:
            action, logits = self._select_action(obs)
            new_obs, reward, terminated, truncated, info = self.env.step(action)
            exp = (obs, action, reward, logits.squeeze(0))
            self.buffer.append(exp)
            episode_logits.append(logits)
            obs = new_obs
            if truncated or terminated:
                break
        return torch.cat(episode_logits)

    def _calc_loss(self, logits, gamma_rewards):
        log_probs = F.log_softmax(logits, dim=-1)
        n_steps = len(self.actions)
        selected_log_probs = log_probs[torch.arange(n_steps), self.actions]
        loss = -(selected_log_probs * gamma_rewards).sum()
        return loss

    def learn(self, num_episodes):
        for episode in range(num_episodes):
            logits = self._play_episode()

            rewards = self._extract_rewards()
            gamma_rewards = self._gamma_rewards(rewards)
            self._prepare_buffer(gamma_rewards) # s, a, r, s', R

            loss = self._calc_loss(logits, gamma_rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
