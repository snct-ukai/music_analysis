import librosa
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import soundfile as sf

def main(filepath):
    filename = os.path.basename(filepath)

    samplerate = sf.info(filepath).samplerate
    data, sr = librosa.load(filepath, sr = samplerate)
    rms = librosa.feature.rms(y=data)
    zcr = librosa.feature.zero_crossing_rate(y=data)

    stft = librosa.stft(data)
    spectral_flux = np.sqrt(
            np.mean(np.abs(np.diff(np.abs(stft))**2), axis=0)
        )
    
    # normalize
    rms = (rms - np.mean(rms)) / np.std(rms)
    zcr = (zcr - np.mean(zcr)) / np.std(zcr)
    spectral_flux = (spectral_flux - np.mean(spectral_flux)) / np.std(spectral_flux)

    # diff
    rms_diff = np.abs(np.diff(rms[0]))
    zcr_diff = np.abs(np.diff(zcr[0]))
    spectral_flux_diff = np.abs(np.diff(spectral_flux))

    # plot
    fig, ax = plt.subplots(6, 1, figsize=(10, 20))
    ax[0].plot(rms[0], label='RMS')
    ax[0].set_title('RMS')
    ax[0].set_xlabel('frame')
    ax[0].set_ylabel('RMS')

    ax[1].plot(rms_diff, label='RMS diff')
    ax[1].set_title('RMS diff')
    ax[1].set_xlabel('frame')
    ax[1].set_ylabel('RMS diff')

    ax[2].plot(zcr[0], label='ZCR')
    ax[2].set_title('ZCR')
    ax[2].set_xlabel('frame')
    ax[2].set_ylabel('ZCR')

    ax[3].plot(zcr_diff, label='ZCR diff')
    ax[3].set_title('ZCR diff')
    ax[3].set_xlabel('frame')
    ax[3].set_ylabel('ZCR diff')

    ax[4].plot(spectral_flux, label='Spectral Flux')
    ax[4].set_title('Spectral Flux')
    ax[4].set_xlabel('frame')
    ax[4].set_ylabel('Spectral Flux')

    ax[5].plot(spectral_flux_diff, label='Spectral Flux diff')
    ax[5].set_title('Spectral Flux diff')
    ax[5].set_xlabel('frame')
    ax[5].set_ylabel('Spectral Flux diff')

    plt.tight_layout()
    save_dir = "./output/feat_visualize"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"feat_visualize_{filename}.png")
    plt.savefig(save_path, dpi=600)

if __name__ == "__main__":
    dir_path = sys.argv[1]
    for filename in os.listdir(dir_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(dir_path, filename)
            main(filepath)