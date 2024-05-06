import numpy as np
import banpei

def sst(data : np.ndarray, window_size : int):
    """
    data : np.ndarray
        1D array
    window_size : int
        window size
    """
    s = banpei.SST(w=window_size)
    return s.detect(data)