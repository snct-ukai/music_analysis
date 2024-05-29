import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 音楽ファイルを読み込む
filepath = filename = 'A:/graduationResearch/analyze/wav/02 我逢人 [Remastered 2020].wav'
y, sr = librosa.load(filepath)

# ビートをトラックする
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# オンセットを検出する
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

# ビートとオンセットを組み合わせて小節の頭を特定する
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

plt.figure(figsize=(14, 5))
plt.plot(np.arange(len(y)) / sr , y)
plt.vlines(beat_times, -1, 1, color='r', alpha=0.75, linestyle='--', label='Beats')
plt.vlines(onset_times, -1, 1, color='g', alpha=0.75, linestyle='--', label='Onsets')
plt.legend()
plt.grid(True)
plt.show()