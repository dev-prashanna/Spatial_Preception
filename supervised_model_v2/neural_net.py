import torch
import torch.nn as nn

class RobotNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(3,32),
            nn.ReLU(),

            nn.Linear(32,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,2),
            nn.Tanh()
        )
    def forward(self,x):
        return self.net(x)
    
   
model=RobotNet()

try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print("model loaded")

except:
    print("no model found, using random weights")

def predict(model,sensor_values):
        with torch.no_grad():
            x = torch.tensor(sensor_values, dtype=torch.float32)
            y = model(x).numpy()
        return float(y[0]), float(y[1])
