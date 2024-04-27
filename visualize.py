import librosa
from librosa.feature.rhythm import tempo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import window_function as wf
import anomaly_detection as ad
from concurrent.futures import ThreadPoolExecutor

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=None) for audioFile in audioFiles]

# plot audio waveforms and rms on the waveforms for each audio file at a sheet
fig, axs = plt.subplots(8, len(audioData), figsize=(20, 20))
for i, (y, sr) in enumerate(audioData):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    y = librosa.effects.trim(y)[0]
    TEMPO = tempo(onset_envelope=onset_env, sr=sr)[0]
    playtime = y.size / sr
    window_size = 1000 #int(60 * len(y) / (TEMPO * playtime))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    print("calc chroma complete")

    zcr : np.ndarray
    zcr_sr : np.ndarray
    mfcc : np.ndarray
    spectral_rolloff : np.ndarray
    spectral_rolloff_sr : np.ndarray
    rms : np.ndarray
    rms_sr : np.ndarray
    spectral_centroid : np.ndarray
    spectral_centroid_sr : np.ndarray
    spectral_bandwidth : np.ndarray
    spectral_bandwidth_sr : np.ndarray

    def calc_zcr():
        global zcr
        global zcr_sr
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr = np.nan_to_num(zcr)
        zcr_sr = ad.sst(zcr, window_size)
        zcr_sr = zcr_sr / np.max(zcr_sr)
        print("calc zcr complete")
    
    def calc_mfcc():
        global mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        print("calc mfcc complete")

    def calc_spectral_rolloff():
        global spectral_rolloff
        global spectral_rolloff_sr
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff = spectral_rolloff / np.max(spectral_rolloff)
        spectral_rolloff = np.nan_to_num(spectral_rolloff)
        spectral_rolloff_sr = ad.sst(spectral_rolloff, window_size)
        spectral_rolloff_sr = spectral_rolloff_sr / np.max(spectral_rolloff_sr)
        print("calc spectral_rolloff complete")

    def calc_rms():
        global rms
        global rms_sr
        rms = librosa.feature.rms(y=y)[0]
        rms = np.nan_to_num(rms)
        rms_sr = ad.sst(rms, window_size)
        rms_sr = rms_sr / np.max(rms_sr)
        print("calc rms complete")

    def calc_spectral_centroid():
        global spectral_centroid
        global spectral_centroid_sr
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid = spectral_centroid / np.max(spectral_centroid)
        spectral_centroid = np.nan_to_num(spectral_centroid)
        spectral_centroid_sr = ad.sst(spectral_centroid, window_size)
        spectral_centroid_sr = spectral_centroid_sr / np.max(spectral_centroid_sr)
        print("calc spectral_centroid complete")

    def calc_spectral_bandwidth():
        global spectral_bandwidth
        global spectral_bandwidth_sr
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_bandwidth = spectral_bandwidth / np.max(spectral_bandwidth)
        spectral_bandwidth = np.nan_to_num(spectral_bandwidth)
        spectral_bandwidth_sr = ad.sst(spectral_bandwidth, window_size)
        spectral_bandwidth_sr = spectral_bandwidth_sr / np.max(spectral_bandwidth_sr)
        print("calc spectral_bandwidth complete")

    with ThreadPoolExecutor() as executor:
        executor.submit(calc_zcr)
        executor.submit(calc_mfcc)
        executor.submit(calc_spectral_rolloff)
        executor.submit(calc_rms)
        executor.submit(calc_spectral_centroid)
        executor.submit(calc_spectral_bandwidth)

    #calc_zcr()
    #calc_mfcc()
    #calc_spectral_rolloff()
    #calc_rms()
    #calc_spectral_centroid()
    #calc_spectral_bandwidth()

    axs[0].set_title("audio waveform")
    axs[0].plot(np.arange(len(y)) / sr, y)
    axs[1].set_title("chroma")
    axs[1].imshow(chroma, aspect='auto', origin='lower', cmap='viridis')
    axs[2].set_title("Zero Crossing Rate")
    axs[2].plot(np.arange(len(zcr)) / len(zcr) * len(y) / sr, zcr)
    axs[2].plot(np.arange(len(zcr_sr)) / len(zcr_sr) * len(y) / sr, zcr_sr)
    #axs[3].set_title("Fundamental Frequency")
    #axs[3].plot(np.arange(len(f0)) / len(f0) * len(y) / sr, f0)
    axs[3].set_title("MFCC")
    axs[3].imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    axs[4].set_title("spectral_rolloff")
    axs[4].plot(np.arange(len(spectral_rolloff)) / len(spectral_rolloff) * len(y) / sr, spectral_rolloff)
    axs[4].plot(np.arange(len(spectral_rolloff_sr)) / len(spectral_rolloff_sr) * len(y) / sr, spectral_rolloff_sr)
    axs[5].set_title("rms")
    axs[5].plot(np.arange(len(rms)) / len(rms) * len(y) / sr, rms)
    axs[5].plot(np.arange(len(rms_sr)) / len(rms_sr) * len(y) / sr, rms_sr)
    #axs[8].set_title("pitch")
    #axs[8].imshow(pitches, aspect='auto', origin='lower', cmap='viridis')
    axs[6].set_title("spectral_centroid")
    axs[6].plot(np.arange(len(spectral_centroid)) / len(spectral_centroid) * len(y) / sr, spectral_centroid)
    axs[6].plot(np.arange(len(spectral_centroid_sr)) / len(spectral_centroid_sr) * len(y) / sr, spectral_centroid_sr)
    axs[7].set_title("spectral_bandwidth")
    axs[7].plot(np.arange(len(spectral_bandwidth)) / len(spectral_bandwidth) * len(y) / sr, spectral_bandwidth)
    axs[7].plot(np.arange(len(spectral_bandwidth_sr)) / len(spectral_bandwidth_sr) * len(y) / sr, spectral_bandwidth_sr)
    
plt.tight_layout()
plt.show()