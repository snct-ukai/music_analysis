import numpy as np
import pandas as pd

def window_function(data : np.ndarray, window_size : int):
    """
    data : np.ndarray
        1D array
    window_size : int
        window size
    """
    s = pd.Series(data)
    return s.rolling(window_size, center=True).mean().values