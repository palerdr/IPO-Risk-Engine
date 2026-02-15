import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_classes: int):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.ReLU = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.ReLU(x)
        x = self.linear_2(x)
        return x
    