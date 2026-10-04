import torch.nn.functional as F
from torch import nn


class ffnn(nn.Module):
    """Linear -> ReLU -> Linear. From From-Algorithm-to-RTL/examples/ffnn."""

    def __init__(self, input_size=64, hidden_size=48, output_size=4):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))
