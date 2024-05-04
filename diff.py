import numpy as np

def diff(x: np.ndarray, h: int) -> np.ndarray:
    res = -x[4:] + 8*x[3:-1] - 8*x[1:-3] + x[:-4]
    return res/(12*h)