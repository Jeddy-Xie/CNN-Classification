import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*12*12,30)
        self.fc2 = nn.Linear(30, 10)


    def forward(self, x):
        # Convolve input with 5 filters of size 5x5, apply ReLU nonlinearity, and pool
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Flatten the output of the convolution layer
        x = x.view(x.size(0), -1)
        # Apply a fully connected layer with 30 units and ReLU nonlinearity
        x = nn.functional.relu(self.fc1(x))
        # Apply a fully connected layer with 10 units
        x = self.fc2(x)
        return x