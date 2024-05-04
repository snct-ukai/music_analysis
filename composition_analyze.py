import composition_detector
import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=4800) for audioFile in audioFiles]

y, sr = audioData[0]

composition_change_index, range = composition_detector.composition_detector(y, sr)
plt.plot(np.arange(len(y)) / sr, y)
for index in composition_change_index:
    plt.axvline(x=index * range / sr, color='r')

plt.show()