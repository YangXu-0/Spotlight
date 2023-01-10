import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, clean_data, noisy_data):
        'Initialization'
        self.clean_data = clean_data
        self.noisy_data = noisy_data
        # Don't need labels

    def __len__(self):
        'Returns total number of samples'
        return len(self.clean_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        clean_sample = self.clean_data[index]
        noisy_sample = self.noisy_data[index]

        return clean_sample, noisy_sample
