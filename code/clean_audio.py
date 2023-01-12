import torch
import torch.nn as nn
import numpy as np
import librosa
import scipy
from torch.utils.data import DataLoader

from networks import Model_1
from dataset import Dataset

###### Due to repetitive nature of code, only train_model.py is fully commented ######

# Setting device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Defining directories
audio_dir = '../assets/data/Noisy/full_test.txt'

# Create dataloader
audio = np.loadtxt(audio_dir)
size_input = len(audio[0])
audio = torch.tensor(audio).to(torch.float32)
audio_loader = Dataset(audio, audio)

params = { # hyp
    'batch_size': 1,
    'shuffle': True,
}

audio_loader = DataLoader(audio_loader, **params)

# Denoising
autoencoder = Model_1(size_input, size_input)
# hyp
loss_fn = torch.nn.MSELoss() 
lr = 0.001 
wd = 0.00001 

checkpoint = torch.load('../assets/models/version_1.txt')
autoencoder.load_state_dict(checkpoint['model_state_dict'])

frequencies, counter = [], 0
autoencoder.eval()
with torch.set_grad_enabled(False):
    for _, local_noisy in audio_loader:
        local_noisy = local_noisy.to(device)

        decoded = autoencoder(local_noisy)

        frequencies = frequencies + decoded.tolist()

        counter += 1
        print(counter)

# Turn frequencies back to audio (temp set sr=10000 for testing)
inverse = np.real_if_close(np.fft.ifft(frequencies))
scipy.io.wavfile.write("../assets/data/Denoised/full_test_denoised.wav", 10000, inverse.astype(np.float32))
