import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    def __init__(self, net, env, optimizer=optim.Adam, gamma=0.99, lr=1e-3):
        self.net = net
        self.env = env
        self.gamma = gamma
        self.optimizer = optimizer(net.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.logits = []

    def _select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
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

    def _prepare_buffer(self, gamma_rewards):
        # states уже list[Tensor], делаем batch
        states = torch.stack(self.states, dim=0)  # [T, *obs_shape]
        actions = torch.tensor(self.actions, dtype=torch.long)  # [T]
        returns = torch.tensor(gamma_rewards, dtype=torch.float32)  # [T]
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.unsqueeze(1)  # [T,1]
        return states, actions, returns

    def _extract_rewards(self):
        rewards = [row[2] for row in self.buffer]
        return rewards

    def _play_episode(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.logits.clear()

        obs, info = self.env.reset()
        while True:
            action, logits = self._select_action(obs)
            self.states.append(torch.as_tensor(obs, dtype=torch.float32))
            self.actions.append(action)
            self.logits.append(logits.squeeze(0))

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.rewards.append(reward)

            if truncated or terminated:
                break

        all_logits = torch.cat([l.unsqueeze(0) for l in self.logits], dim=0)
        return all_logits

    def _calc_loss(self, logits, returns, actions):
        log_probs = F.log_softmax(logits, dim=-1)
        T = logits.shape[0]

        # actions: [T], returns: [T,1]
        selected = log_probs[torch.arange(T), actions]  # → [T]

        # loss: -(sum over t) return_t * log π(a_t|s_t)
        loss = -(selected * returns.squeeze(1)).sum()
        return loss

    def learn(self, num_episodes):
        self.net.train()
        self.episode_rewards = []
        for episode in range(num_episodes):
            all_logits = self._play_episode()

            total_reward = sum(self.rewards)  # 1) суммируем reward из буфера
            self.episode_rewards.append(total_reward)  # 2) сохраняем в историю
            print(f"Episode {episode + 1}\tTotal reward: {total_reward:.2f}")

            gamma_rs = self._gamma_rewards(self.rewards)
            states, actions, returns = self._prepare_buffer(gamma_rs) # s, a, R

            loss = self._calc_loss(all_logits, returns, actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def play(self, render=False):
        self.net.eval()

        obs, info = self.env.reset()
        total_reward = 0.0
        if render:
            self.env.render()

        while True:
            with torch.no_grad():
                state_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits  = self.net(state_t)
                probs   = F.softmax(logits, dim=-1)
                action  = torch.multinomial(probs, num_samples=1).item()

            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if render:
                self.env.render()

            if terminated or truncated:
                break

        self.net.train()
        return total_reward
