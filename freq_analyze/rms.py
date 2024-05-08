import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.004)
    rms = librosa.feature.rms(y=y)[0]
    rms = rms / np.max(rms)
    rms_sst = sst(rms, window_size)
    rms_sst = rms_sst / np.max(rms_sst)
    return [rms, rms_sst]