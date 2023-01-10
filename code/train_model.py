import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from networks import Model_1 # Importing model
from dataset import Dataset  # Importing dataset

# Setting device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

################# Load Training/Testing Data #################
# Defining directories
training_clean_dir = '../assets/data/Clean/training_clean.txt'
training_noisy_dir = '../assets/data/Noisy/training_noisy.txt'

testing_clean_dir = '../assets/data/Clean/testing_clean.txt'
testing_noisy_dir = '../assets/data/Noisy/testing_noisy.txt'

data_dirs = [[training_clean_dir, training_noisy_dir], \
             [testing_clean_dir, testing_noisy_dir]]

# Create dataloaders
datasets = []
for i in range(len(data_dirs)):
    # Load preprocessed data
    temp_clean = np.loadtxt(data_dirs[i][0])
    temp_noisy = np.loadtxt(data_dirs[i][1])

    size_input = len(temp_clean[0]) # inefficient, I know

    # Turn into tensors
    temp_clean = torch.tensor(temp_clean).to(torch.float32) # defaults to float64
    temp_noisy = torch.tensor(temp_noisy).to(torch.float32)

    # Create dataset object
    temp_obj = Dataset(temp_clean, temp_noisy)

    # Save object to list
    datasets.append(temp_obj)

params = { # hyp
    'batch_size': 64,
    'shuffle': True,
}
max_epochs = 50

training_loader = DataLoader(datasets[0], **params)
testing_loader = DataLoader(datasets[1], **params)

#################### Training and Validation ####################
autoencoder = Model_1(size_input, size_input) # defining model

# hyp
loss_fn = torch.nn.MSELoss() # defining loss function
lr = 0.001 # defining learning rate
wd = 0.00001 # defining weight decay

# defining optimizer function
optim = torch.optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=wd)

training_loss, validation_loss = [], []
for epoch in range(max_epochs):
    # Training
    autoencoder.train() # set to training mode
    counter = 0
    for local_clean, local_noisy in training_loader:
        # Transfer to device
        local_clean, local_noisy = local_clean.to(device), local_noisy.to(device)

        # Encode-decode
        decoded = autoencoder(local_noisy)

        # Evaluate loss
        loss = loss_fn(decoded, local_clean)

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Process loss
        counter += 1
        print(f'Partial training loss: {loss.data} ({counter})')
    training_loss.append(loss.detach().cpu().numpy())

    # Validation
    autoencoder.eval() # set to testing mode
    counter = 0
    with torch.set_grad_enabled(False):
        for local_clean, local_noisy in testing_loader:
            # Transfer to device
            local_clean, local_noisy = local_clean.to(device), local_noisy.to(device)

            # Encode-decode
            decoded = autoencoder(local_noisy)

            # Evaluate loss
            loss = loss_fn(decoded, local_clean)

            # Process loss
            counter += 1
            print(f'Partial validation loss: {loss.data} ({counter})')
        validation_loss.append(loss.detach().cpu().numpy())  

print(f"The mean training loss was: {np.mean(training_loss)}")
print(f"The mean validation loss was: {np.mean(validation_loss)}")

##################### Exporting the model ####################
torch.save({
    'epoch': epoch,
    'model_state_dict': autoencoder.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss
}, '../assets/models/version_2.txt')
