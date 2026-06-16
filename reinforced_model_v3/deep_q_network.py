import torch.nn as nn


class QNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(6,128),
            nn.LeakyReLU(0.01),

            nn.Linear(128,128),
            nn.LeakyReLU(0.01),

            nn.Linear(128,64),
            nn.LeakyReLU(0.01),

            nn.Linear(64,3)
        )

    def forward(self,x):
        return self.network(x)