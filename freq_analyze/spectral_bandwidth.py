import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.004)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth = spectral_bandwidth / np.max(spectral_bandwidth)
    spectral_bandwidth_sst = sst(spectral_bandwidth, window_size)
    spectral_bandwidth_sst = spectral_bandwidth_sst / np.max(spectral_bandwidth_sst)
    return spectral_bandwidth_sst