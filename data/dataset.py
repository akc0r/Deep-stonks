import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
from .labeling import smooth_labels

class LOBDataset(Dataset):
    def __init__(self, data_dir, train=True, k=10, T=100, alpha=0.00002):
        """
        Args:
            data_dir (str): Path to the dataset directory (e.g. .../NoAuction_DecPre_Training).
            train (bool): If True, use training files, else testing.
            k (int): Horizon for labeling.
            T (int): History window size (number of past events).
            alpha (float): Threshold for labeling.
        """
        self.T = T
        self.k = k
        self.train = train
        
        file_lists = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        if not file_lists:
            raise ValueError(f"No .txt files found in {data_dir}")
            
        data_pieces = []
        
        print(f"Loading {len(file_lists)} files from {data_dir}...")
        
        for fpath in file_lists:
            raw_data = np.loadtxt(fpath).T
        
            lob_data = raw_data[:, :40]
            
            ask_price_1 = lob_data[:, 0]
            bid_price_1 = lob_data[:, 2]
            mid_prices = (ask_price_1 + bid_price_1) / 2.0
            
            labels = smooth_labels(mid_prices, k, alpha)
            
            data_pieces.append((lob_data, labels))
            
        self.features = np.concatenate([d[0] for d in data_pieces], axis=0)
        self.labels = np.concatenate([d[1] for d in data_pieces], axis=0)
        
        if train:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)
            self.std[self.std < 1e-8] = 1.0

        self.features = (self.features - self.mean) / self.std
        
        self.valid_indices = np.arange(self.T, len(self.features) - self.k)
        
        self.features = torch.FloatTensor(self.features)
        
        self.labels = torch.LongTensor(self.labels + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        
        window = self.features[t-self.T : t]
        
        window = window.unsqueeze(0)
        
        label = self.labels[t-1]
        
        return window, label
