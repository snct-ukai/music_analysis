import numpy as np

def cos_sim(x : np.ndarray, y : np.ndarray) -> float:
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def time_series_cos_sim(d: np.ndarray) -> np.ndarray:
    res = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        res[i] = cos_sim(d[0], d[i])
    return res