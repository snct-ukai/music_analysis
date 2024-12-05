import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import wave
import csv
import matplotlib

def feature_visualize(y, y_array) -> None :
    hop_length = 512
    frame_length = 2048
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    zero_crossings = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    spectral_flux = np.sqrt(np.mean(np.diff(np.abs(stft), axis=1)**2, axis=0))

    section_length = [0]
    for y_ in y_array:
        section_length.append(len(y_))
    section_length = np.cumsum(section_length)
    
    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(4, 1, 1)
    for i, y_ in enumerate(y_array):
        ax1.plot(np.arange(section_length[i], section_length[i + 1]), y_, label=f'Section {i + 1}', c=matplotlib.cm.tab20(i))
    ax1.set_title('Waveform')
    ax1.tick_params(bottom=False)
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(energy)
    ax2.set_title('Energy')
    ax2.tick_params(bottom=False)
    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(zero_crossings)
    ax3.set_title('Zero Crossings')
    ax3.tick_params(bottom=False)
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(spectral_flux)
    ax4.set_title('Spectral Flux')
    ax4.tick_params(bottom=False)
    plt.tight_layout()
    plt.savefig(f'{dirname}/features.png', format='png', dpi=600)

if __name__ == "__main__":
    try:
        dirname = sys.argv[1]
        filename = "music.wav"
        filePath = os.path.join(dirname, filename)
        with wave.open(filePath, "rb") as wf:
            sampleRate = wf.getframerate()
        data, sr = librosa.load(filePath, sr=sampleRate)
        y_array = []
        section_dir = os.path.join(dirname, "section")
        fileNames = [i for i in csv.reader(open(os.path.join(section_dir, "list.csv"))).__next__()]
        for file in fileNames:
            y, _ = librosa.load(os.path.join(section_dir, file))
            y_array.append(y)
        feature_visualize(data, y_array)
    except IndexError:
        print("Usage: python feature_visualizer.py <audio file path>")
        sys.exit(1)