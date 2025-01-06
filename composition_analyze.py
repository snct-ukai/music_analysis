import composition_detector
import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from freq_analyze import mfcc
from util import anomaly_detection
import sys

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=4800) for audioFile in audioFiles]

y, sr = librosa.load(sys.argv[1], sr=4800)

y, composition_change_index, time_range = composition_detector.composition_detector(y, sr)

#fig, axs = plt.subplots(1, 1, figsize=(20, 20))

segments = composition_change_index * time_range / sr
import soundfile as sf
import os
sr = sf.info(sys.argv[1]).samplerate
y, _ = librosa.load(sys.argv[1], sr=sr)
output_dir = f'./output/sst/{sys.argv[1].split("/")[-1].split(".")[0]}'
os.makedirs(output_dir, exist_ok=True)
for i in range(len(segments) - 1):
    output_file_name = os.path.join(output_dir, f'{i}.wav')
    sf.write(output_file_name, y[int(segments[i] * sr):int(segments[i + 1] * sr)], sr)

plt.plot(np.arange(len(y)) / sr, y)
for seg in segments:
    plt.axvline(x=seg, color='r')

plt.show()