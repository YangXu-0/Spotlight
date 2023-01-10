import torch
import torch.nn as nn

# Unsupervised learning approach w/ a denoising autoencoder

class Model_1(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        # Layers
        self.stack = nn.Sequential(
            nn.Linear(in_size, in_size),   # input layer
            nn.Linear(in_size, in_size),   # encoder layer
            nn.BatchNorm1d(in_size),       # batch normalization layer
            nn.LeakyReLU(),                # activation layer
            nn.Linear(in_size, in_size),   # middle hidden layer
            nn.Linear(in_size, in_size),   # decoder layer
            nn.BatchNorm1d(in_size),       # batch normalization layer
            nn.LeakyReLU(),                # activation layer
            nn.Linear(in_size, out_size)   # output layer
        )

    def forward(self, x):
        # Forward step
        logits = self.stack(x)
        return logits
