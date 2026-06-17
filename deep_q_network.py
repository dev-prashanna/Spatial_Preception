import torch.nn as nn


class QNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(7,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),

            nn.Linear(256,256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),

            nn.Linear(256,128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.01),

            nn.Linear(128,3)
        )

    def forward(self,x):
        return self.network(x)