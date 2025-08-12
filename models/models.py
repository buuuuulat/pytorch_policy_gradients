import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_relu_seq = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=out_features)
        )

    def forward(self, x):
        return self.fc_relu_seq(x)


class A2CNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Common part
        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
        )

        # Actor Head
        self.actor_head = nn.Linear(in_features=128, out_features=out_features)

        # Critic Head
        self.critic_head = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        return self.actor_head(shared_features), self.critic_head(shared_features)
