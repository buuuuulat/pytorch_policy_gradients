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