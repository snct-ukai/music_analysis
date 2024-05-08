import numpy as np
import librosa
from util.anomaly_detection import sst

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.004)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid = spectral_centroid / np.max(spectral_centroid)
    spectral_centroid = np.nan_to_num(spectral_centroid)
    spectral_centroid_sst = sst(spectral_centroid, window_size)
    spectral_centroid_sst = spectral_centroid_sst / np.max(spectral_centroid_sst)
    return spectral_centroid_sst