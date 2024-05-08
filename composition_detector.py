import librosa
from librosa.feature.rhythm import tempo
import numpy as np
from util import window_function as wf, anomaly_detection as ad, diff
from util.filter import low_pass_filter
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Tuple
from freq_analyze import normalize_wave as nw, zcr, rms as rm, spectral_rolloff as sro, spectral_centroid as sc, spectral_bandwidth as sb

def composition_detector(y : np.ndarray, sr : int) -> Tuple[np.ndarray, float]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    y = librosa.effects.trim(y)[0]
    beat : np.ndarray = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
    beat_time = librosa.frames_to_time(beat, sr=sr)
    normalize_wave = np.zeros(y.size)
    window_size = int(sr * 0.004)

    future_list = []

    with ThreadPoolExecutor() as executor:
        normalize_wave = executor.submit(nw.calc, y, sr, beat_time)
        future_list.append(normalize_wave)
        zcr_sst = executor.submit(zcr.calc, y, sr)
        future_list.append(zcr_sst)
        spectral_rolloff_sst = executor.submit(sro.calc, y, sr)
        future_list.append(spectral_rolloff_sst)
        rms_res = executor.submit(rm.calc, y, sr)
        future_list.append(rms_res)
        spectral_centroid_sst = executor.submit(sc.calc, y, sr)
        future_list.append(spectral_centroid_sst)
        spectral_bandwidth_sst = executor.submit(sb.calc, y, sr)
        future_list.append(spectral_bandwidth_sst)

        _ = futures.wait(future_list)

    normalize_wave = normalize_wave.result()
    zcr_sst = zcr_sst.result()
    spectral_rolloff_sst = spectral_rolloff_sst.result()
    rms_res = rms_res.result()
    spectral_centroid_sst = spectral_centroid_sst.result()
    spectral_bandwidth_sst = spectral_bandwidth_sst.result()

    rms = rms_res[0]
    rms_sst = rms_res[1]

    for i in range(zcr_sst.size):
        zcr_sst[i] *= normalize_wave[int(i * normalize_wave.size // zcr_sst.size)] * rms[int(i * rms.size // zcr_sst.size)]
    for i in range(spectral_rolloff_sst.size):
        spectral_rolloff_sst[i] *= normalize_wave[int(i * normalize_wave.size // spectral_rolloff_sst.size)] * rms[int(i * rms.size // spectral_rolloff_sst.size)]
    for i in range(rms_sst.size):
        rms_sst[i] *= normalize_wave[int(i * normalize_wave.size // rms_sst.size)] * rms[int(i * rms.size // rms_sst.size)]
    for i in range(spectral_centroid_sst.size):
        spectral_centroid_sst[i] *= normalize_wave[int(i * normalize_wave.size // spectral_centroid_sst.size)] * rms[int(i * rms.size // spectral_centroid_sst.size)]
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