import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from models import A2CNet
from agents import A2CAgent
from experience import A2CBuffer


# Parameters
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LEARNING_RATE = 3e-4
NUM_EPISODES = 5000
ENTROPY_COEF = 0.01
UNROLL_STEPS = 0 # If > 0, uses unrolling n episodes
VALUE_COEF = 0.5
GRAD_CLIP = 0.5

env = gym.make(ENV_NAME)
policy = A2CNet(in_features=env.observation_space.shape[0], out_features=env.action_space.n)
buffer = A2CBuffer(gamma=GAMMA)
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=policy.parameters())
agent = A2CAgent(policy, optimizer, VALUE_COEF, GRAD_CLIP)
writer = SummaryWriter(comment='_A2C')


def train(n_episodes, unroll_steps=0, normalize_advantages=True):
    """
    Train Logic
    :param n_episodes: Number of episodes to be played
    :param unroll_steps: If True, plays only first n steps
    :param normalize_advantages: If True, advantages are normalized
    """
    policy.train()
    episode_rewards = []
    for episode in range(n_episodes):
        obs, info = env.reset()
        buffer.clear()
        ep_return = 0.0
        steps_in_ep = 0.0

        last_loss = last_entropy = last_policy_loss = last_value_loss = None
        last_adv_mean = None

        terminated = truncated = False
        while not (terminated or truncated):
            action, logits, value = agent.select_action(obs)
            new_obs, reward, terminated, truncated, info = env.step(action)
            buffer.append([obs, action, reward, logits, value])

            ep_return += reward
            steps_in_ep += 1
            obs = new_obs

            boundary = (unroll_steps and (steps_in_ep % unroll_steps == 0))

            if boundary or terminated or truncated:
                with torch.no_grad():
                    if terminated:
                        bootstrap_value = 0.0
                    else:
                        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                        _, v_boot = policy(obs_t)
                        bootstrap_value = float(v_boot.squeeze())

        returns = buffer.calculate_discounted_rewards(bootstrap_value)

        logits, actions, returns, values = buffer.prepare_buffer(returns, normalize_returns=False)
        advantages = (returns - values).detach()
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss, entropy, policy_loss, value_loss = agent.update_grads(logits, advantages, actions, values,
                                                                    returns, ENTROPY_COEF)

        last_loss = loss.item()
        last_entropy = entropy.item()
        last_policy_loss = policy_loss.item()
        last_value_loss = value_loss.item()
        last_adv_mean = advantages.mean().item()

        buffer.clear()

        episode_rewards.append(ep_return)
        print(f"Episode {episode + 1}\tTotal reward: {ep_return:.2f}")

        # Logging
        if episode % 20 == 0:
            writer.add_scalar("Loss", last_loss if last_loss is not None else 0.0, episode)
            mean10 = sum(episode_rewards[-10:]) / max(1, min(10, len(episode_rewards)))
            writer.add_scalar("Mean_10_Reward", mean10, episode)
            writer.add_scalar("Episode_Reward", ep_return, episode)
            writer.add_scalar("Steps_per_Episode", steps_in_ep, episode)
            writer.add_scalar("Mean_Advantage_LastUpdate", last_adv_mean if last_adv_mean is not None else 0.0, episode)
            writer.add_scalar("Entropy", last_entropy if last_entropy is not None else 0.0, episode)
            writer.add_scalar("Policy_Loss", last_policy_loss if last_policy_loss is not None else 0.0, episode)
            writer.add_scalar("Value_Loss", last_value_loss if last_value_loss is not None else 0.0, episode)
    writer.flush()

train(NUM_EPISODES, UNROLL_STEPS)