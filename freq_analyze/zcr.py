import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.004)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr = zcr / np.max(zcr)
    zcr_sst = sst(zcr, window_size)
    zcr_sst = zcr_sst / np.max(zcr_sst)
    return zcr_sst