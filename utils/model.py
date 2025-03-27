import torch
import torch.nn as nn
from torchvision import models

class PlateRecognitionModel(nn.Module):
    def __init__(self, num_chars):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(256, 128, bidirectional=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(256, num_chars + 1)  # +1 for CTC blank token

    def forward(self, x):
        x = self.cnn(x)
        x = x.unsqueeze(1).repeat(1, 30, 1)  # 30 timesteps for LSTM
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x
