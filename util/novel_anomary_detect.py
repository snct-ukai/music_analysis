from util import anomaly_detection
import novelty_function
import numpy as np
import sys
import matplotlib.pyplot as plt
import librosa
import ruptures as rpt

def novelty_detection(data : np.ndarray, window_size : int):
    """
    data : np.ndarray
        1D array
    window_size : int
        window size
    """
    spectral_novelty = novelty_function.freq_novelty_function(data)
    energy_novelty = novelty_function.rms_novelty_function(data)
    novelty = spectral_novelty + energy_novelty
    novelty = novelty / np.max(novelty)
    mean = np.convolve(novelty, np.ones(window_size) / window_size, mode="same")
    std = np.sqrt(np.convolve((novelty - mean) ** 2, np.ones(window_size) / window_size, mode="same"))
    threshold = mean + 4 * std
    result = np.where(novelty > threshold)[0]
    return result, novelty



if __name__ == "__main__":
    try:
        filePath = sys.argv[1]
        y, sr = librosa.load(filePath)
        y = librosa.effects.trim(y)[0]
        window_size = 1024
        result, novelty = novelty_detection(y, window_size)
        print(result)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(novelty)
        #for i in range(len(result) - 1):
        #    ax.axvline(result[i], color="red")
        plt.show()

    except IndexError:
        print("Usage: python novel_anomary_detect.py <audio file path>")
        sys.exit(1)