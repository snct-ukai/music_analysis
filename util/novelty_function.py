import numpy as np
import librosa

def freq_novelty_function(data) -> np.ndarray :
    Y = np.log(np.abs(librosa.stft(data) + 1e-9))
    spectral_novelty = None
    n_freq, n_tf = Y.shape
    spectral_novelty = np.zeros(n_tf - 1)
    for f in range(0, n_freq):
        tmp = Y[f, 1:] - Y[f, :-1]
        tmp[tmp < 0.0] = 0.0
        spectral_novelty += tmp
    spectral_novelty /= np.max(spectral_novelty)
    return spectral_novelty

def rms_novelty_function(data) -> np.ndarray:
    mean_square = librosa.feature.rms(y=data)[0]
    energy_novelty = np.log(mean_square[1:] / (mean_square[:-1] + 1e-9) + 1e-9)
    energy_novelty[energy_novelty < 0.0] = 0.0
    energy_novelty /= np.max(energy_novelty)
    return energy_novelty

def onset(data) -> np.ndarray:
    onset_env = librosa.onset.onset_strength(y=data, sr=22050)
    onset_env = np.log(onset_env + 1e-9)
    onset_env /= np.max(onset_env)
    return onset_env

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    filePath = sys.argv[1]
    y, sr = librosa.load(filePath)
    y = librosa.effects.trim(y)[0]
    spectral_novelty = freq_novelty_function(y)
    energy_novelty = rms_novelty_function(y)
    harmonics, percussive = librosa.effects.hpss(y)
    spectral_novelty_harmonics = freq_novelty_function(harmonics)
    spectral_novelty_percussive = freq_novelty_function(percussive)
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax1.plot(spectral_novelty)
    ax1.set_title("Spectral Novelty Function")
    ax2 = fig.add_subplot(512)
    ax2.plot(energy_novelty)
    ax2.set_title("Energy Novelty Function")
    ax3 = fig.add_subplot(513)
    ax3.plot(spectral_novelty_harmonics)
    ax3.set_title("Spectral Novelty Function (Harmonics)")
    ax4 = fig.add_subplot(514)
    ax4.plot(spectral_novelty_percussive)
    ax4.set_title("Spectral Novelty Function (Percussive)")
    ax5 = fig.add_subplot(515)
    onset_env = onset(y)
    ax5.plot(onset_env)
    ax5.set_title("Onset Function")
    
    plt.tight_layout()

    plt.show()