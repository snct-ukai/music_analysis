import numpy as np
import librosa
from util.anomaly_detection import sst
from util.cos_sim import time_series_cos_sim

def calc(y : np.ndarray, sr : float) -> np.ndarray:
    window_size = int(sr * 0.005)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cos_sim = time_series_cos_sim(chroma.T) + 1
    chroma_cos_sim_sst = sst(chroma_cos_sim, window_size)
    chroma_cos_sim_sst = (chroma_cos_sim_sst - np.mean(chroma_cos_sim_sst)) / np.std(chroma_cos_sim_sst)
    return chroma_cos_sim_sst