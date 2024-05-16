import numpy as np
import librosa
from util.anomaly_detection import sst
from util.cos_sim import time_series_cos_sim
from typing import Tuple

def calc(y : np.ndarray, sr : float) -> Tuple[np.ndarray, np.ndarray]:
    window_size = int(sr * 0.005)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    mfcc_cos_sim = time_series_cos_sim(mfcc.T) + 1
    mfcc_cos_sim = mfcc_cos_sim / np.max(mfcc_cos_sim)
    mfcc_cos_sim_sst = sst(mfcc_cos_sim, window_size)
    mfcc_cos_sim_sst = (mfcc_cos_sim_sst - np.mean(mfcc_cos_sim_sst)) / np.std(mfcc_cos_sim_sst)
    return [mfcc_cos_sim_sst, mfcc]