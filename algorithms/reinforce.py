import gymnasium as gym
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from models import Net
from agents import PGAgent
from experience import PGExperienceBuffer


# Parameters
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LEARNING_RATE = 1e-3

env = gym.make(ENV_NAME)
policy = Net(in_features=env.observation_space.shape[0], out_features=env.action_space.n)
buffer = PGExperienceBuffer(gamma=GAMMA)
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=policy.parameters())
agent = PGAgent(policy, optimizer)


def train(n_episodes):
    policy.train()
    episode_rewards = []
    for episode in range(n_episodes):
        done = False
        step_idx = 0
        obs, info = env.reset()
        while not done:
            action, logits = agent.select_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            buffer.append([obs, action, reward, logits])
            obs = new_obs
            step_idx += 1

            if terminated or truncated:
                break

        returns = buffer.calculate_discounted_rewards()
        logits, actions, returns = buffer.prepare_buffer(returns, normalize_returns=True)
        loss = agent.update_grads(logits, returns, actions)

        total_reward = sum(buffer.rewards)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}\tTotal reward: {total_reward:.2f}")
        buffer.clear()
train(5000)