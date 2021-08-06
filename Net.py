import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Linear(128, 2)
    
    def forward(self, x):
        return self.net(x)
