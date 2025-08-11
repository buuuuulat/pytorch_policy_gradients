import torch
import torch.nn.functional as F


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

    def update_grads(self, logits, returns, actions):
        """
        Loss Calculation and gradient update: L = -sum(Selected Log Probs * Returns)
        :param logits: Tensor: [n_steps, n_actions]
        :param returns: Tensor: [n_steps, 1]
        :param actions: Tensor: [n_steps]
        :return: Loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        T = logits.shape[0]
        # actions: [T], returns: [T,1]
        selected = log_probs[torch.arange(T), actions]  # → [T]
        # loss: -(sum over t) return_t * log π(a_t|s_t)
        loss = -(selected * returns.squeeze(1)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
