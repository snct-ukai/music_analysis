import composition_detector
import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from freq_analyze import mfcc
from util import anomaly_detection

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=4800) for audioFile in audioFiles]

y, sr = audioData[0]

y, composition_change_index, time_range = composition_detector.composition_detector(y, sr)

#fig, axs = plt.subplots(1, 1, figsize=(20, 20))

plt.plot(np.arange(len(y)) / sr, y)
for index in composition_change_index:
    plt.axvline(x=index * time_range / sr, color='r')

plt.show()