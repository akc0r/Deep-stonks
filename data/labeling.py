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
    
    future_means = np.convolve(mid_prices, np.ones(k)/k, mode='valid')
    
    cumsum = np.cumsum(mid_prices)
    
    m_minus = (cumsum[k:] - cumsum[:-k]) / k
    
    current_prices = mid_prices[:-k]
    
    changes = (m_minus - current_prices) / current_prices
    
    labels_raw = np.zeros_like(changes, dtype=int)
    labels_raw[changes > alpha] = 1
    labels_raw[changes < -alpha] = -1

    full_labels = np.zeros(T, dtype=int)
    full_labels[:-k] = labels_raw
    
    return full_labels
