import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.004)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_rolloff = spectral_rolloff / np.max(spectral_rolloff)
    spectral_rolloff = np.nan_to_num(spectral_rolloff)
    spectral_rolloff_sst = sst(spectral_rolloff, window_size)
    spectral_rolloff_sst = spectral_rolloff_sst / np.max(spectral_rolloff_sst)
    return spectral_rolloff_sst