import librosa
import numpy as np
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib as mpl

def main(filepath):
    sr = sf.info(filepath).samplerate
    y, sr = librosa.load(filepath, sr=sr)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    S = np.abs(librosa.stft(y))
    spectral_flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) / S.shape[0]

    # Convert frames to time in seconds
    rms_time = librosa.frames_to_time(np.arange(rms.shape[1]), sr=sr)
    chroma_time = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr)
    spectral_centroid_time = librosa.frames_to_time(np.arange(spectral_centroid.shape[1]), sr=sr)
    spectral_bandwidth_time = librosa.frames_to_time(np.arange(spectral_bandwidth.shape[1]), sr=sr)
    spectral_flatness_time = librosa.frames_to_time(np.arange(spectral_flatness.shape[1]), sr=sr)
    spectral_rolloff_time = librosa.frames_to_time(np.arange(spectral_rolloff.shape[1]), sr=sr)
    zero_crossing_rate_time = librosa.frames_to_time(np.arange(zero_crossing_rate.shape[1]), sr=sr)
    spectral_flux_time = librosa.frames_to_time(np.arange(spectral_flux.shape[0]), sr=sr)

    # 余白を0にする
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0

    fig, ax = plt.subplots(4, 2, figsize=(15, 15))
    ax[0, 0].plot(rms_time, rms[0])
    ax[0, 0].set_title('RMS')
    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('RMS')

    ax[0, 1].imshow(chroma, aspect='auto', origin='lower', cmap='coolwarm', extent=[chroma_time.min(), chroma_time.max(), 0, chroma.shape[0]])
    ax[0, 1].set_title('Chroma')
    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel('Chroma')

    ax[1, 0].plot(spectral_centroid_time, spectral_centroid[0])
    ax[1, 0].set_title('Spectral Centroid')
    ax[1, 0].set_xlabel('Time [s]')
    ax[1, 0].set_ylabel('Spectral Centroid [Hz]')

    ax[1, 1].plot(spectral_bandwidth_time, spectral_bandwidth[0])
    ax[1, 1].set_title('Spectral Bandwidth')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Spectral Bandwidth [Hz]')

    ax[2, 0].plot(spectral_flatness_time, spectral_flatness[0])
    ax[2, 0].set_title('Spectral Flatness')
    ax[2, 0].set_xlabel('Time [s]')
    ax[2, 0].set_ylabel('Spectral Flatness')

    ax[2, 1].plot(spectral_rolloff_time, spectral_rolloff[0])
    ax[2, 1].set_title('Spectral Rolloff')
    ax[2, 1].set_xlabel('Time [s]')
    ax[2, 1].set_ylabel('Spectral Rolloff [Hz]')

    ax[3, 0].plot(zero_crossing_rate_time, zero_crossing_rate[0])
    ax[3, 0].set_title('Zero Crossing Rate')
    ax[3, 0].set_xlabel('Time [s]')
    ax[3, 0].set_ylabel('Zero Crossing Rate')

    ax[3, 1].plot(spectral_flux_time, spectral_flux)
    ax[3, 1].set_title('Spectral Flux')
    ax[3, 1].set_xlabel('Time [s]')
    ax[3, 1].set_ylabel('Spectral Flux')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()

if __name__ == '__main__':
    main(sys.argv[1])