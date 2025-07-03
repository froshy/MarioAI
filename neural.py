import torch
import torch.nn as nn

class MNet(nn.Module):
    
    def __init__(self, in_channels, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride = 1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU()
        )
        
        self.head = nn.Linear(512, num_actions)
    
    def forward(self, x):
        x = self.conv(x/255)
        x = self.fc(x)
        return self.head(x)