import torch
import torch.nn as nn

class DeepLOB(nn.Module):
    def __init__(self, y_len=3):
        super().__init__()
        self.y_len = y_len
        
        # Convolutional Block 1
        # Input: (N, 1, 100, 40)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'), # Using 'same' padding to keep T=100
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        
        # Convolutional Block 2
        # Input: (N, 16, 100, 20)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        
        # Convolutional Block 3
        # Input: (N, 16, 100, 10)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10)), # Aggregates features to 1
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, (4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        # Output of conv3: (N, 16, 100, 1)
        
        # Inception Module
        self.inception = InceptionModule(input_channels=16, out_channels=32)
        # Output: (N, 128, 100, 1)
        
        # LSTM
        # Input to LSTM: (N, SequenceLength, Features)
        # We need to reshape (N, 128, 100, 1) -> (N, 100, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(64, y_len)
        
    def forward(self, x):
        # x: (N, 1, 100, 40)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.inception(x)
        
        # Reshape for LSTM: (N, 128, 100, 1) -> (N, 100, 128)
        # Squeeze dim 3 (features=1)
        x = x.squeeze(3)
        # Permute to (N, 100, 128)
        x = x.permute(0, 2, 1)
        
        # LSTM
        # output, (hn, cn)
        _, (hn, _) = self.lstm(x)
        # hn: (num_layers, N, hidden_size) -> (1, N, 64)
        x = hn[-1] # Take last layer hidden state
        
        x = self.fc(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        
        # Branch 1: 1x1 conv
        # Note: In PyTorch Conv2d, kernel (H, W). Here (Time, Features). features dim is 1.
        # So we act on Time. Kernel (1, 1).
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(out_channels)
        )
        
        # Branch 2: 1x1 -> 3x1 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels)
        )
        
        # Branch 3: 5x1 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels)
        )
        
        # Branch 4: MaxPool -> Conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(input_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Mix on channel dimension (dim 1)
        return torch.cat([b1, b2, b3, b4], dim=1)
