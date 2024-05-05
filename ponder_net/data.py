import torch
from torch.utils.data import Dataset

class ParityDataset(Dataset):
    def __init__(self, n_samples, n_elements):
        self.n_samples = n_samples
        self.n_elements = n_elements

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.zeros((self.n_elements,))

        num_non_zero = torch.randint(1, self.n_elements+1, (1,)).item()

        x[:num_non_zero] = torch.randint(0, 2, (num_non_zero,)) * 2 - 1
        x = x[torch.randperm(self.n_elements)]

        y = (x == 1.).sum() % 2

        return x, y
