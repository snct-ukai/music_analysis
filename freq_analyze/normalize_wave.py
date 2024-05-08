import numpy as np
from concurrent import futures

def calc(y : np.ndarray, sr : float, beat_time : np.ndarray) -> np.ndarray:
    normalize_wave = np.zeros(y.size)
    x = np.arange(y.size)

    def normalize_wave_sub(i : int) -> np.ndarray:
        sigma = sr * (beat_time[i + 1] - beat_time[i]) / 8
        myu = beat_time[i] * sr
        return np.exp(-(((x - myu) / sigma) ** 2) / 2)
    
    future_list = []
    with futures.ThreadPoolExecutor(max_workers=24) as executor:
        for i in range(len(beat_time) - 1):
            future = executor.submit(normalize_wave_sub, i)
            future_list.append(future)
        _ = futures.wait(future_list)

    for future in future_list:
        normalize_wave += future.result()
    return normalize_wave