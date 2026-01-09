import numpy as np

def smooth_labels(mid_prices, k, alpha=0.00002):
    """
    Generates labels based on the smoothing method (Zhang et al., 2018).
    
    Args:
        mid_prices (np.array): Array of mid-prices (T,).
        k (int): Prediction horizon.
        alpha (float): Threshold for stationary class.
        
    Returns:
        np.array: Labels {-1, 0, 1} mapped to {0, 1, 2} or similar.
                  Here we return {-1, 0, 1} as requested, but usually for CrossEntropy we need 0, 1, 2.
                  Prompt says: Up (+1), Down (-1), Stationary (0).
                  We will map: -1 -> 0, 0 -> 1, 1 -> 2 for PyTorch CrossEntropy compatibility.
    """
    T = len(mid_prices)
    labels = np.zeros(T, dtype=int)
    
    # Calculate mean of future mid-prices m_minus(t) over horizon k
    # m_minus(t) = 1/k * sum_{i=1}^k p_{t+i}
    # We can use convolution to compute moving average efficiently
    
    future_means = np.convolve(mid_prices, np.ones(k)/k, mode='valid')
    # np.convolve 'valid' returns length N - K + 1. 
    # expected result at index t corresponds to window [t, t+k-1].
    # But we want future [t+1, t+k].
    
    # Let's do it manually or carefully align.
    # m_minus[t] (paper notation) is mean of p[t+1]...p[t+k]
    
    # Using pandas rolling if available is easier, but numpy is preferred for cleanliness.
    # Cumsum is fast.
    cumsum = np.cumsum(mid_prices)
    # mean[t] over k future steps: (cumsum[t+k] - cumsum[t]) / k
    # indices: t from 0 to T-k-1
    
    m_minus = (cumsum[k:] - cumsum[:-k]) / k
    
    # We compare m_minus[t] with p[t] (current price)
    # The array m_minus starts at t=0 (using p[1]..p[k]?) NO.
    # cumsum[k] - cumsum[0] = p[1] + ... + p[k]. This is future relative to t=0.
    
    # Align shapes
    # p_t should be mid_prices[:-k]
    
    current_prices = mid_prices[:-k]
    
    # l_t = (m_minus[t] - p_t) / p_t
    changes = (m_minus - current_prices) / current_prices
    
    # Apply threshold
    labels_raw = np.zeros_like(changes, dtype=int)
    labels_raw[changes > alpha] = 1   # Up
    labels_raw[changes < -alpha] = -1 # Down
    # Stationary is 0 (default)
    
    # Pad the end labels with 0 or exclude them
    # We will pad with 0s to match original length T, though strictly they are unknown.
    full_labels = np.zeros(T, dtype=int)
    full_labels[:-k] = labels_raw
    
    return full_labels
