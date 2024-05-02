import librosa
from librosa.feature.rhythm import tempo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import window_function as wf
import anomaly_detection as ad
from concurrent.futures import ThreadPoolExecutor

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=4800) for audioFile in audioFiles]

# plot audio waveforms and rms on the waveforms for each audio file at a sheet
fig, axs = plt.subplots(9, len(audioData), figsize=(20, 20))
for i, (y, sr) in enumerate(audioData):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    y = librosa.effects.trim(y)[0]
    TEMPO = tempo(onset_envelope=onset_env, sr=sr)[0]
    beat : np.ndarray = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[1]
    beat_time = librosa.frames_to_time(beat, sr=sr)
    normalize_wave = np.zeros(y.size)
    playtime = y.size / sr
    window_size = 20
    chord_wave = np.zeros(y.size)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print("calc chroma complete")

    zcr : np.ndarray
    zcr_sst : np.ndarray
    mfcc : np.ndarray
    mfcc_cos_sim : np.ndarray
    spectral_rolloff : np.ndarray
    spectral_rolloff_sst : np.ndarray
    rms : np.ndarray
    rms_sst : np.ndarray
    spectral_centroid : np.ndarray
    spectral_centroid_sst : np.ndarray
    spectral_bandwidth : np.ndarray
    spectral_bandwidth_sst : np.ndarray
    chroma_sst : np.ndarray = np.zeros((12, len(y)))

    def calc_normalize_wave():
        global normalize_wave
        normalize_wave = np.zeros(y.size)
        
        x = np.arange(y.size)
        for j in range(len(beat_time) - 1):
            sigma = sr * (beat_time[j + 1] - beat_time[j]) / 9
            myu = beat_time[j] * sr
            normalize_wave += np.exp(-(((x - myu) / sigma) ** 2) / 2)

    def calc_zcr():
        global zcr
        global zcr_sst
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr = np.nan_to_num(zcr)
        zcr_sst = ad.sst(zcr, window_size)
        zcr_sst = zcr_sst / np.max(zcr_sst)
        print("calc zcr complete")
    
    def calc_mfcc():
        global mfcc
        global mfcc_cos_sim
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        print("calc mfcc complete")

    def calc_spectral_rolloff():
        global spectral_rolloff
        global spectral_rolloff_sst
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff = spectral_rolloff / np.max(spectral_rolloff)
        spectral_rolloff = np.nan_to_num(spectral_rolloff)
        spectral_rolloff_sst = ad.sst(spectral_rolloff, window_size)
        spectral_rolloff_sst = spectral_rolloff_sst / np.max(spectral_rolloff_sst)
        print("calc spectral_rolloff complete")

    def calc_rms():
        global rms
        global rms_sst
        rms = librosa.feature.rms(y=y)[0]
        rms = np.nan_to_num(rms)
        rms_sst = ad.sst(rms, window_size)
        rms_sst = rms_sst / np.max(rms_sst)
        print("calc rms complete")

    def calc_spectral_centroid():
        global spectral_centroid
        global spectral_centroid_sst
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = spectral_centroid / np.max(spectral_centroid)
        spectral_centroid = np.nan_to_num(spectral_centroid)
        spectral_centroid_sst = ad.sst(spectral_centroid, window_size)
        spectral_centroid_sst = spectral_centroid_sst / np.max(spectral_centroid_sst)
        print("calc spectral_centroid complete")

    def calc_spectral_bandwidth():
        global spectral_bandwidth
        global spectral_bandwidth_sst
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth = spectral_bandwidth / np.max(spectral_bandwidth)
        spectral_bandwidth = np.nan_to_num(spectral_bandwidth)
        spectral_bandwidth_sst = ad.sst(spectral_bandwidth, window_size)
        spectral_bandwidth_sst = spectral_bandwidth_sst / np.max(spectral_bandwidth_sst)
        print("calc spectral_bandwidth complete")
    
    def calc_chroma_sst(i : int):
        global chroma_sst
        chroma_sst[i] = wf.window_function(chroma[i], window_size)
        chroma_sst[i] = chroma_sst / np.max(chroma_sst)
        print("calc chroma_sst complete")

    with ThreadPoolExecutor() as executor:
        executor.submit(calc_normalize_wave)
        executor.submit(calc_zcr)
        executor.submit(calc_mfcc)
        executor.submit(calc_spectral_rolloff)
        executor.submit(calc_rms)
        executor.submit(calc_spectral_centroid)
        executor.submit(calc_spectral_bandwidth)
    wave_sum = zcr_sst + spectral_rolloff_sst + rms_sst + spectral_centroid_sst + spectral_bandwidth_sst
    wave_sum_sst = ad.sst(wave_sum, 10)
    
    for i in range(zcr_sst.size):
        zcr_sst[i] *= normalize_wave[i * normalize_wave.size // zcr_sst.size]
    for i in range(spectral_rolloff_sst.size):
        spectral_rolloff_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_rolloff_sst.size]
    for i in range(rms_sst.size):
        rms_sst[i] *= normalize_wave[i * normalize_wave.size // rms_sst.size]
    for i in range(spectral_centroid_sst.size):
        spectral_centroid_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_centroid_sst.size]
    for i in range(spectral_bandwidth_sst.size):
        spectral_bandwidth_sst[i] *= normalize_wave[i * normalize_wave.size // spectral_bandwidth_sst.size]
    

    axs[0].set_title("audio waveform")
    axs[0].plot(np.arange(len(y)) / sr, y)
    for b in beat_time:
        axs[0].axvline(b, color='r')

    axs[1].set_title("chroma")
    axs[1].imshow(chroma, aspect='auto', origin='lower', cmap='viridis')

    axs[2].set_title("Zero Crossing Rate")
    axs[2].plot(np.arange(len(zcr)) / len(zcr) * len(y) / sr, zcr)
    axs[2].plot(np.arange(len(zcr_sst)) / len(zcr_sst) * len(y) / sr, zcr_sst)

    axs[3].set_title("MFCC")
    axs[3].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')

    axs[4].set_title("spectral_rolloff")
    axs[4].plot(np.arange(len(spectral_rolloff)) / len(spectral_rolloff) * len(y) / sr, spectral_rolloff)
    axs[4].plot(np.arange(len(spectral_rolloff_sst)) / len(spectral_rolloff_sst) * len(y) / sr, spectral_rolloff_sst)

    axs[5].set_title("rms")
    axs[5].plot(np.arange(len(rms)) / len(rms) * len(y) / sr, rms)
    axs[5].plot(np.arange(len(rms_sst)) / len(rms_sst) * len(y) / sr, rms_sst)

    axs[6].set_title("spectral_centroid")
    axs[6].plot(np.arange(len(spectral_centroid)) / len(spectral_centroid) * len(y) / sr, spectral_centroid)
    axs[6].plot(np.arange(len(spectral_centroid_sst)) / len(spectral_centroid_sst) * len(y) / sr, spectral_centroid_sst)

    axs[7].set_title("spectral_bandwidth")
    axs[7].plot(np.arange(len(spectral_bandwidth)) / len(spectral_bandwidth) * len(y) / sr, spectral_bandwidth)
    axs[7].plot(np.arange(len(spectral_bandwidth_sst)) / len(spectral_bandwidth_sst) * len(y) / sr, spectral_bandwidth_sst)

    axs[8].set_title("wave_sum")
    axs[8].plot(np.arange(len(wave_sum)) / len(wave_sum) * len(y) / sr, wave_sum)
    axs[8].plot(np.arange(len(wave_sum_sst)) / len(wave_sum_sst) * len(y) / sr, wave_sum_sst)
    
plt.tight_layout()
plt.show()