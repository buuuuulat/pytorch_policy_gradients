import numpy as np
import torch


class PGExperienceBuffer:
    """
    Experience buffer class: manages sars and logits for Policy Gradients Methods
    """
    def __init__(self, gamma=0.99):
        """
        :param gamma: Discount coefficient
        """
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.logits = []

    def append(self, exp):
        """
        :param exp: List: [state, action, reward, logits]
        :return: None
        """
        self.states.append(exp[0])
        self.actions.append(exp[1])
        self.rewards.append(exp[2])
        self.logits.append(exp[3])

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logits = []

    def calculate_discounted_rewards(self):
        """
        Calculates sum of maximum discounted rewards from current step till the end for every step
        :return: List of discounted rewards
        """
        res = []
        sum_r = 0.0
        for r in reversed(self.rewards):
            sum_r = sum_r * self.gamma + r
            res.append(sum_r)
        return list(reversed(res))

    def prepare_buffer(self, returns, normalize_returns=False):
        """
        Prepares torch tensors and normalizes rewards
        :param returns: List of discounted rewards
        :param normalize_returns: Check True if normalization is needed
        :return: Torch Tensors of states, actions, returns
        """
        #states = torch.stack(self.states, dim=0)  # [T, *obs_shape]
        actions = torch.tensor(self.actions, dtype=torch.long)  # [T]
        logits = torch.cat([l.unsqueeze(0) for l in self.logits], dim=0)
        logits = logits.squeeze(1)
        returns = torch.tensor(returns, dtype=torch.float32)  # [T]
        if normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = returns.unsqueeze(1)  # [T,1]
        return logits, actions, returns

