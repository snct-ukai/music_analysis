import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.005)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff_sst = sst(spectral_rolloff, window_size)
    spectral_rolloff_sst = (spectral_rolloff_sst - np.mean(spectral_rolloff_sst)) / np.std(spectral_rolloff_sst)
    return spectral_rolloff_sst