import random, struct, sys, os
import numpy as np
from torch.utils.data import Dataset

class LorenzDataset(Dataset):
    def __init__(self, args, which='train'):
        self.args = args
        if(which == 'train'):
            self.dat = np.load("data/data.npy")
        if(which == 'valid'):
            self.dat = np.load("data/data_valid.npy")
        if(which == 'test'):
            self.dat = np.load("data/data_test.npy")

        self.N = self.dat.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        states = self.dat[idx]
        data = states[0:self.args.input_len, 0:self.args.state_dim]
        label = states[self.args.input_len:self.args.input_len + self.args.output_len, 0:self.args.state_dim_out]
        return data.astype(float), label.astype(float)
