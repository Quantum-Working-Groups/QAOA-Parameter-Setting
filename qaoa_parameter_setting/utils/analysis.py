
"""Methods to analyse data."""

import numpy as np


def quantile(func_counts: dict, alpha: float):
    """Compute a quantile from the counts."""
    x_vals = sorted(func_counts.keys())
    y_vals = np.cumsum([func_counts[x] for x in x_vals])
    idx = np.argmin(abs(y_vals - alpha))
    return x_vals[idx]

def mean_obj(func_counts: dict):
    """Return the mean of the counts"""
    return sum([k*v for k, v in func_counts.items()])

def standard_error_mean(counts: dict, num_shots: int):
    """Get the standard error on the mean."""
    mean = mean_obj(counts)
    return np.sqrt(sum([prob * (val - mean)**2 for val, prob in counts.items()])/ num_shots)
