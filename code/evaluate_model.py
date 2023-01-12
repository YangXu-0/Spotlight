import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from networks import Model_1 # Importing model
from dataset import Dataset  # Importing dataset

###### Due to repetitive nature of code, only train_model.py is fully commented ######

# Setting device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Defining directories
testing_clean_dir = '../assets/data/Clean/validation_clean.txt'
testing_noisy_dir = '../assets/data/Noisy/validation_noisy.txt'

# Create dataloaders
testing_clean = np.loadtxt(testing_clean_dir)
testing_noisy = np.loadtxt(testing_noisy_dir)

size_input = len(testing_clean[0])

testing_clean = torch.tensor(testing_clean).to(torch.float32)
testing_noisy = torch.tensor(testing_noisy).to(torch.float32)

testing_set = Dataset(testing_clean, testing_noisy)
params = { # hyp
    'batch_size': 1,
    'shuffle': True,
}

testing_loader = DataLoader(testing_set, **params)

# Calculate loss
autoencoder = Model_1(size_input, size_input)

# hyp
loss_fn = torch.nn.MSELoss() # defining loss function
lr = 0.001 # defining learning rate
wd = 0.00001 # defining weight decay

# defining optimizer function
optim = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wd)

# loading in the model
checkpoint = torch.load('../assets/models/version_1.txt')
autoencoder.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])

testing_loss = []
counter = 0
autoencoder.eval()
with torch.set_grad_enabled(False):
    for local_clean, local_noisy in testing_loader:
        local_clean, local_noisy = local_clean.to(device), local_noisy.to(device)

        decoded = autoencoder(local_noisy)

        loss = loss_fn(decoded, local_clean)

        counter += 1
        print(f'Partial validation loss: {loss.data} ({counter})')
        testing_loss.append(loss.detach().cpu().numpy())
print(f"The mean testing loss was: {np.mean(testing_loss)}")
