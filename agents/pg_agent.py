import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_



class PGAgent:
    """
    Agent class: chooses actions and updates grads
    """
    def __init__(self, policy, optimizer):
        """
        :param policy: PyTorch nn policy
        :param optimizer: torch.optim optimizer
        """
        self.policy = policy
        self.optimizer = optimizer

    def select_action(self, state):
        """
        Selects action from probability distribution
        :param state: Observation state from the environment
        :return: Selected action and logits
        """
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)

        action = torch.multinomial(probs, num_samples=1).item()
        return action, logits

    def update_grads(self, logits, returns, actions, entropy_coef=0.01):
        """
        Loss Calculation and gradient update: L = -sum(Selected Log Probs * Returns)
        :param logits: Tensor: [n_steps, n_actions]
        :param returns: Tensor: [n_steps, 1]
        :param actions: Tensor: [n_steps]
        :param entropy_coef: If greater than 0, turns on the entropy bonus
        :return: Loss
        """
        log_probs = F.log_softmax(logits, dim=-1) # log π(a|s)
        probs = torch.exp(log_probs)

        T = logits.shape[0]
        # Actions: [T], returns: [T,1]
        selected = log_probs[torch.arange(T), actions]  # → [T]

        # Policy Gradient Loss: -(sum over t) return_t * log π(a_t|s_t)
        pg_loss = -(selected * returns.squeeze(1)).sum()

        # Entropy bonus
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = pg_loss - entropy * entropy_coef

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, entropy
    

class A2CAgent(PGAgent):
    def __init__(self, policy, optimizer, value_coef=0.3, max_grad_norm=0.1):
        super().__init__(policy, optimizer)
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def select_action(self, state):
        """
        Selects action from probability distribution
        :param state: Observation state from the environment
        :return: Selected action and logits
        """
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.policy(state)
        probs = F.softmax(logits, dim=-1)

        action = torch.multinomial(probs, num_samples=1).item()
        return action, logits, value

    def update_grads(self, logits, advantage, actions, values, returns, entropy_coef=0.01):
        """
        Loss Calculation and gradient update: L = -sum(Selected Log Probs * Advantage)
        :param logits: Tensor: [n_steps, n_actions]
        :param advantage: Tensor: [n_steps, 1]
        :param actions: Tensor: [n_steps]
        :param entropy_coef: If greater than 0, turns on the entropy bonus
        :return: Loss
        """
        log_probs = F.log_softmax(logits, dim=-1) # log π(a|s)
        probs = torch.exp(log_probs)

        T = logits.shape[0]
        # Actions: [T], advantage: [T]
        selected = log_probs[torch.arange(T), actions]  # → [T]

        # Policy Loss
        policy_loss = -(selected * advantage.squeeze(1)).mean()

        # Value Loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = -(probs * log_probs).sum(dim=1).mean()

        loss = policy_loss + self.value_coef * value_loss - entropy * entropy_coef

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.detach(), entropy.detach(), policy_loss.detach(), value_loss.detach()


class A2CMultiEnvAgent(A2CAgent):
    def __init__(self, policy, optimizer, value_coef=0.3, max_grad_norm=0.1):
        super().__init__(policy, optimizer, value_coef, max_grad_norm)

    def select_action(self, obs_np):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        logits, values = self.policy(obs_t)  # [N, A], [N, 1]
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()  # [N]
        return actions.cpu().numpy(), logits, values
