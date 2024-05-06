import librosa
from librosa.feature.rhythm import tempo
import numpy as np
from util import window_function as wf, anomaly_detection as ad, diff
from util.filter import low_pass_filter
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

def composition_detector(y : np.ndarray, sr : int) -> Tuple[np.ndarray, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    y = librosa.effects.trim(y)[0]
    beat : np.ndarray = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
    beat_time = librosa.frames_to_time(beat, sr=sr)
    normalize_wave = np.zeros(y.size)
    window_size = int(sr * 0.004)

    def calc_normalize_wave() -> np.ndarray:
        normalize_wave = np.zeros(y.size)
        x = np.arange(y.size)
        for j in range(len(beat_time) - 1):
            sigma = sr * (beat_time[j + 1] - beat_time[j]) / 8
            myu = beat_time[j] * sr
            normalize_wave += np.exp(-(((x - myu) / sigma) ** 2) / 2)
        return normalize_wave

    def calc_zcr_sst() -> np.ndarray:
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr = np.nan_to_num(zcr)
        zcr_sst = ad.sst(zcr, window_size)
        zcr_sst = zcr_sst / np.max(zcr_sst)
        return zcr_sst

    def calc_spectral_rolloff_sst() -> np.ndarray:
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff = spectral_rolloff / np.max(spectral_rolloff)
        spectral_rolloff = np.nan_to_num(spectral_rolloff)
        spectral_rolloff_sst = ad.sst(spectral_rolloff, window_size)
        spectral_rolloff_sst = spectral_rolloff_sst / np.max(spectral_rolloff_sst)
        return spectral_rolloff_sst

    def calc_rms_sst() -> np.ndarray:
        rms = librosa.feature.rms(y=y)[0]
        rms = rms / np.max(rms)
        rms = np.nan_to_num(rms)
        rms_sst = ad.sst(rms, window_size)
        rms_sst = rms_sst / np.max(rms_sst)
        return [rms, rms_sst]

    def calc_spectral_centroid_sst() -> np.ndarray:
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = spectral_centroid / np.max(spectral_centroid)
        spectral_centroid = np.nan_to_num(spectral_centroid)
        spectral_centroid_sst = ad.sst(spectral_centroid, window_size)
        spectral_centroid_sst = spectral_centroid_sst / np.max(spectral_centroid_sst)
        return spectral_centroid_sst

    def calc_spectral_bandwidth_sst() -> np.ndarray:
        global spectral_bandwidth
        global spectral_bandwidth_sst
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth = spectral_bandwidth / np.max(spectral_bandwidth)
        spectral_bandwidth = np.nan_to_num(spectral_bandwidth)
        spectral_bandwidth_sst = ad.sst(spectral_bandwidth, window_size)
        spectral_bandwidth_sst = spectral_bandwidth_sst / np.max(spectral_bandwidth_sst)
        return spectral_bandwidth_sst

    with ThreadPoolExecutor() as executor:
        normalize_wave = executor.submit(calc_normalize_wave)
        zcr_sst = executor.submit(calc_zcr_sst)
        spectral_rolloff_sst = executor.submit(calc_spectral_rolloff_sst)
        rms_res = executor.submit(calc_rms_sst)
        spectral_centroid_sst = executor.submit(calc_spectral_centroid_sst)
        spectral_bandwidth_sst = executor.submit(calc_spectral_bandwidth_sst)
    
    normalize_wave = normalize_wave.result()
    zcr_sst = zcr_sst.result()
    spectral_rolloff_sst = spectral_rolloff_sst.result()
    rms_res = rms_res.result()
    spectral_centroid_sst = spectral_centroid_sst.result()
    spectral_bandwidth_sst = spectral_bandwidth_sst.result()

    rms = rms_res[0]
    rms_sst = rms_res[1]

    for i in range(zcr_sst.size):
        zcr_sst[i] *= normalize_wave[i * normalize_wave.size // zcr_sst.size] * rms[i * rms.size // zcr_sst.size]
    for i in range(spectral_rolloff_sst.size):
        spectral_rolloff_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_rolloff_sst.size] * rms[i * rms.size // spectral_rolloff_sst.size]
    for i in range(rms_sst.size):
        rms_sst[i] *= normalize_wave[i * normalize_wave.size // rms_sst.size] * rms[i * rms.size // rms_sst.size]
    for i in range(spectral_centroid_sst.size):
        spectral_centroid_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_centroid_sst.size] * rms[i * rms.size // spectral_centroid_sst.size]
    for i in range(spectral_bandwidth_sst.size):
        spectral_bandwidth_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_bandwidth_sst.size] * rms[i * rms.size // spectral_bandwidth_sst.size]
    
    ### メロディ変化点検出
    # 各パラメータの変化点検出結果を合成
    wave_sum = zcr_sst + spectral_rolloff_sst + rms_sst + spectral_centroid_sst + spectral_bandwidth_sst
    wave_sum = wave_sum / np.max(wave_sum) # 正規化
    wave_sum_sst = ad.sst(wave_sum, window_size // 2) # 合成した変化点検出結果に対して異常検知
    # ビートごとに山が来る正規分布の列で作られた波との積を取る
    for i in range(wave_sum_sst.size):
        wave_sum_sst[i] *= normalize_wave[i * normalize_wave.size // wave_sum_sst.size]
    wave_sum_sst = low_pass_filter(wave_sum_sst, sr) # ローパスフィルタ
    wave_sum_sst = wave_sum_sst / np.max(wave_sum_sst) * 2 
    wave_sum_sst = np.exp(wave_sum_sst) # 指数関数で変化点を増幅
    wave_sum_sst = wave_sum_sst / np.max(wave_sum_sst) # 正規化

    delta_wave_sum_sst = diff.diff(wave_sum_sst, 1 / sr)
    delta_wave_sum_sst[delta_wave_sum_sst < 0] = 0
    delta_wave_sum_sst = delta_wave_sum_sst / np.max(delta_wave_sum_sst)

    change_point_index = np.argwhere(delta_wave_sum_sst > 0.1)
    change_point_index = change_point_index.flatten()
    return [change_point_index, y.size / delta_wave_sum_sst.size]