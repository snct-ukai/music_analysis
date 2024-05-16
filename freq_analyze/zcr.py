import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.005)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_sst = sst(zcr, window_size)
    zcr_sst = (zcr_sst - np.mean(zcr_sst)) / np.std(zcr_sst)
    return zcr_sst