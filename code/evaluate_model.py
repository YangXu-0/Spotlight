import torch
import torch.nn as nn
import numpy as np

from networks import Model_1 # Importing model
from dataset import Dataset  # Importing dataset

# Setting device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Load data

# Compare predictions with labels

# Output results