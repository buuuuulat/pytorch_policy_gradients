import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from models import A2CNet
from agents import A2CAgent
from experience import A2CBuffer


# Parameters
ENV_NAME = "CartPole-v1"
GAMMA = 0.9
LEARNING_RATE = 1e-3
NUM_EPISODES = 1000
ENTROPY_COEF = 0.01
UNROLL_STEPS = 500 # If > 0, plays only first n episodes
VALUE_COEF = 0.5
GRAD_CLIP = 0.1

env = gym.make(ENV_NAME)
policy = A2CNet(in_features=env.observation_space.shape[0], out_features=env.action_space.n)
buffer = A2CBuffer(gamma=GAMMA)
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=policy.parameters())
agent = A2CAgent(policy, optimizer, VALUE_COEF, GRAD_CLIP)
writer = SummaryWriter(comment='_A2C')


def train(n_episodes, unroll_steps=0):
    """
    Train Logic
    :param n_episodes: Number of episodes to be played
    :param unroll_steps: If True, plays only first n steps
    """
    policy.train()
    episode_rewards = []
    for episode in range(n_episodes):
        done = False
        step_idx = 0
        obs, info = env.reset()
        while not done:
            action, logits, value = agent.select_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            buffer.append([obs, action, reward, logits, value])
            obs = new_obs
            step_idx += 1

            if (unroll_steps and step_idx == unroll_steps) or (terminated or truncated):
                break

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, v_boot = policy(obs_t)
        bootstrap_value = 0.0 if terminated else float(v_boot.squeeze(0))
        returns = buffer.calculate_discounted_rewards(bootstrap_value)

        logits, actions, returns, values = buffer.prepare_buffer(returns, normalize_returns=False)
        advantage = (returns - values).detach()

        loss, entropy, policy_loss, value_loss = agent.update_grads(logits, advantage, actions, values,
                                                                    returns, ENTROPY_COEF)

        total_reward = sum(buffer.rewards)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}\tTotal reward: {total_reward:.2f}")
        buffer.clear()

        # Logging
        if episode % 20 == 0:
            writer.add_scalar("Loss", loss.item(), episode)
            writer.add_scalar("Mean_10_Reward", sum(episode_rewards[-10:]) / 10, episode)
            writer.add_scalar("Episode_Reward", total_reward, episode)
            writer.add_scalar("Steps_per_Episode", actions.shape[0], episode)
            writer.add_scalar("Mean_10_Advantage", sum(advantage[-10:]) / 10, episode)
            writer.add_scalar("Entropy", entropy, episode)
            writer.add_scalar("Policy_Loss", policy_loss.item(), episode)
            writer.add_scalar("Value_Loss", value_loss.item(), episode)
writer.flush()

train(NUM_EPISODES, UNROLL_STEPS)