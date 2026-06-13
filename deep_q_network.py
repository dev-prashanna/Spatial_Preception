import torch 
import torch.nn as nn
from collections import deque
import random
import numpy as np

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(

            nn.Linear(3,16),
            nn.ReLU(),
            nn.Linear(16,3)
        )
    def forward(self,x):
        return self.net(x)
    
   

    