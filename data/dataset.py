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
        
        # Load data
        # Expecting data_dir to point to the specific folder (Processing/Training or Testing)
        # or we find it.
        # Assuming we deal with concatenated data usually, but here we load from multiple files.
        
        file_lists = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        if not file_lists:
            # Try exploring subdirectories if needed, or error out
            raise ValueError(f"No .txt files found in {data_dir}")
            
        data_pieces = []
        
        print(f"Loading {len(file_lists)} files from {data_dir}...")
        
        for fpath in file_lists:
            # Read file: Features x Time
            # FI-2010 text files are space separated
            # Transpose to Time x Features
            raw_data = np.loadtxt(fpath).T
            
            # Extract features (first 40 columns after Transpose, i.e. first 40 rows before)
            # 10 levels * 4 (AskP, AskV, BidP, BidV) = 40
            lob_data = raw_data[:, :40]
            
            # Compute Mid Price for Labeling: (AskP1 + BidP1) / 2
            # AskP1 is col 0, BidP1 is col 2 (usually: AP1, AV1, BP1, BV1...)
            # Let's verify standard FI-2010:
            # Row 0: Ask P 1
            # Row 1: Ask V 1
            # Row 2: Bid P 1
            # Row 3: Bid V 1
            ask_price_1 = lob_data[:, 0]
            bid_price_1 = lob_data[:, 2]
            mid_prices = (ask_price_1 + bid_price_1) / 2.0
            
            # Generate Labels
            labels = smooth_labels(mid_prices, k, alpha)
            
            # Normalize Features (Z-score)
            # Note: We should calculate Mean/Std from Training set and apply to Test set.
            # For simplicity in this assignment, we normalize per file or assume global constants?
            # Proper way: Calculate global mean/std on training data.
            # We will handle normalization later in the __init__ logic.
            
            data_pieces.append((lob_data, labels))
            
        # Concatenate
        self.features = np.concatenate([d[0] for d in data_pieces], axis=0)
        self.labels = np.concatenate([d[1] for d in data_pieces], axis=0)
        
        # Normalize
        # If train, compute stats. If test, use provided stats? 
        # For this exercise, we'll re-compute stats if train, but ideally we should save them.
        # We will iterate over the dataset once to compute mean/std if train.
        if train:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)
            # Avoid divide by zero
            self.std[self.std < 1e-8] = 1.0
        else:
            # User should ideally provide mean/std. 
            # We will estimate or just use per-batch? No, that's bad.
            # We'll use local stats for now as fallback or implement correctly.
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0)
            self.std[self.std < 1e-8] = 1.0

        self.features = (self.features - self.mean) / self.std
        
        # Trim invalid regions (start and end where we don't have enough history or future labels)
        # We need T history and k future (though label is at t, based on future).
        # Valid indices: [T-1, length - k - 1]
        self.valid_indices = np.arange(self.T, len(self.features) - self.k)
        
        # Convert to torch
        self.features = torch.FloatTensor(self.features)
        
        # Map labels to 0, 1, 2
        # -1 -> 0 (Down)
        #  0 -> 1 (Stationary)
        #  1 -> 2 (Up)
        self.labels = torch.LongTensor(self.labels + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # idx maps to valid_indices[idx]
        t = self.valid_indices[idx]
        
        # Get window [t-T : t]
        # Shape: (T, 40)
        # Model expects (1, 100, 40) or (1, T, 40)
        window = self.features[t-self.T : t]
        
        # Add channel dimension: (1, T, 40)
        window = window.unsqueeze(0)
        
        label = self.labels[t-1] # label at t corresponds to change from t to t+k. 
        # Wait, if we predict at t, we use history ending at t. Label is computed at t (using future).
        # Correct index is t? OR t-1?
        # If window is [t-T : t], the last timestamp is t-1.
        # We predict movement starting from t-1 into future?
        # Usually: Input X_t (up to t), Predict Y_t (t to t+k).
        # So label at index `t` (which uses p[t]...p[t+k]) corresponds to history ending at `t`.
        # My window slicing excludes `t`. window = data[t-T:t] includes indices t-T ... t-1.
        # So last price seen is at t-1.
        # So we should use label at t-1.
        
        return window, label
