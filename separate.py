import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# 音声ファイルの読み込み
audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
filepath = audioFiles[0]
y, sr = librosa.load(filepath)

y = librosa.effects.trim(y)[0]

# テンポとビートの検出
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_frames = librosa.frames_to_samples(beats)

# 1小節の長さを計算
beats_per_measure = 4  # 4拍子として
measure_length = int((tempo / 60) * sr * beats_per_measure)

# フレームごとのエネルギーを計算
hop_length = 512
frame_length = 2048
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])

# ゼロ交差率の計算
zero_crossings = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]

# スペクトルフラックスの計算
stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
spectral_flux = np.sqrt(np.mean(np.diff(np.abs(stft), axis=1)**2, axis=0))

# MFCCの計算
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

# Chroma featureを使用してコード進行を検出
chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

# スペクトルコントラストの計算
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

# トニックフレックストラム（Tonnetz）の計算
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

# メロディックコンターの計算（CQT）
cqt = np.abs(librosa.cqt(y=y, sr=sr, hop_length=hop_length))
melody_contour = np.mean(cqt, axis=0)

# 特徴量の最小長さを取得
min_length = min(len(energy), len(zero_crossings), len(spectral_flux), mfcc.shape[1], chroma.shape[1],
                 spectral_contrast.shape[1], tonnetz.shape[1], len(melody_contour))

# 特徴量のトリミング
energy = energy[:min_length]
zero_crossings = zero_crossings[:min_length]
spectral_flux = spectral_flux[:min_length]
mfcc = mfcc[:, :min_length]
chroma = chroma[:, :min_length]
spectral_contrast = spectral_contrast[:, :min_length]
tonnetz = tonnetz[:, :min_length]
melody_contour = melody_contour[:min_length]

# 特徴量の正規化
energy = (energy - np.mean(energy)) / np.std(energy)
zero_crossings = (zero_crossings - np.mean(zero_crossings)) / np.std(zero_crossings)
spectral_flux = (spectral_flux - np.mean(spectral_flux)) / np.std(spectral_flux)
mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
chroma = (chroma - np.mean(chroma, axis=1, keepdims=True)) / np.std(chroma, axis=1, keepdims=True)
spectral_contrast = (spectral_contrast - np.mean(spectral_contrast, axis=1, keepdims=True)) / np.std(spectral_contrast, axis=1, keepdims=True)
tonnetz = (tonnetz - np.mean(tonnetz, axis=1, keepdims=True)) / np.std(tonnetz, axis=1, keepdims=True)
melody_contour = (melody_contour - np.mean(melody_contour)) / np.std(melody_contour)

energy_diff = np.diff(energy)
zero_crossings_diff = np.diff(zero_crossings)
spectral_flux_diff = np.diff(spectral_flux)
mfcc_diff = np.mean(np.abs(np.diff(mfcc, axis=1)), axis=0)
chroma_diff = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)
spectral_contrast_diff = np.mean(np.abs(np.diff(spectral_contrast, axis=1)), axis=0)
tonnetz_diff = np.mean(np.abs(np.diff(tonnetz, axis=1)), axis=0)
melody_contour_diff = np.abs(np.diff(melody_contour))

# 全ての特徴量の長さを一致させる
min_length = min(len(energy_diff), len(zero_crossings_diff), len(spectral_flux_diff), len(mfcc_diff), len(chroma_diff), len(spectral_contrast_diff), len(tonnetz_diff), len(melody_contour_diff))
energy_diff = energy_diff[:min_length]
zero_crossings_diff = zero_crossings_diff[:min_length]
spectral_flux_diff = spectral_flux_diff[:min_length]
mfcc_diff = mfcc_diff[:min_length]
chroma_diff = chroma_diff[:min_length]
spectral_contrast_diff = spectral_contrast_diff[:min_length]
tonnetz_diff = tonnetz_diff[:min_length]
melody_contour_diff = melody_contour_diff[:min_length]

# 総合的なスコアを計算
combined_score = (
    energy_diff +
    zero_crossings_diff +
    spectral_flux_diff +
    mfcc_diff +
    chroma_diff +
    spectral_contrast_diff +
    tonnetz_diff +
    melody_contour_diff
)

# 局所的な平均と標準偏差を計算
window_size = int(len(combined_score) / (len(y) / sr) * (tempo / 60 * 8))  # 窓サイズ
local_mean = np.convolve(combined_score, np.ones(window_size) / window_size, mode='same')
local_std = np.sqrt(np.convolve((combined_score - local_mean)**2, np.ones(window_size) / window_size, mode='same'))

# 動的な閾値を設定
n = 2.75
dynamic_threshold  = local_mean + n * local_std

# 変化点の検出
change_points = np.where(combined_score > dynamic_threshold)[0]

# 下側の閾値を設定
lower_threshold = local_mean - n * local_std

# 下側の閾値を超える変化点を検出
change_points_below = np.where(combined_score < lower_threshold)[0]
change_points_below_value = lower_threshold[change_points_below] - combined_score[change_points_below]

# 上下の閾値を超える変化点をマージ
change_points = np.sort(np.concatenate([change_points, change_points_below]))

# 始めの2小節は変化点として検出されやすいので除外
change_points = change_points[change_points > 2 * measure_length / hop_length]

change_points_value = np.abs(combined_score[change_points] - local_mean[change_points])

# change_pointsの配列で、もしwindow_sizeにおさまる範囲に複数の候補がある場合、の最大値のみ残す、1フレームずつずらす
for i in range(len(change_points) - window_size):
    if np.all(change_points_value[i:i+window_size] < change_points_value[i+1:i+window_size+1]):
        change_points[i] = 0

# 大きい順に上位50個を取得
change_points = change_points[np.argsort(change_points_value)[::-1]][:50]
change_points = np.sort(change_points)

# 変化点をサンプル単位に変換
change_samples = change_points * hop_length

# 境界点に先頭と末尾を追加
change_samples = np.concatenate(([0], change_samples, [len(y)]))

# 最小セグメント長をサンプル単位で設定
min_segment_length = measure_length * 2

# 短いセグメントを統合するロジック
def should_merge_segments(seg1_start, seg1_end, seg2_start, seg2_end, mfcc, chroma, spectral_contrast, tonnetz, melody_contour, 
                          mfcc_threshold=50, chroma_threshold=0.8, spectral_contrast_threshold=0.5, 
                          tonnetz_threshold=0.5, melody_contour_threshold=0.5):
    mfcc_dist = np.linalg.norm(np.mean(mfcc[:, seg1_start:seg1_end], axis=1) - np.mean(mfcc[:, seg2_start:seg2_end], axis=1))
    chroma_corr = np.corrcoef(np.mean(chroma[:, seg1_start:seg1_end], axis=1), np.mean(chroma[:, seg2_start:seg2_end], axis=1))[0, 1]
    spectral_contrast_corr = np.corrcoef(np.mean(spectral_contrast[:, seg1_start:seg1_end], axis=1), np.mean(spectral_contrast[:, seg2_start:seg2_end], axis=1))[0, 1]
    tonnetz_corr = np.corrcoef(np.mean(tonnetz[:, seg1_start:seg1_end], axis=1), np.mean(tonnetz[:, seg2_start:seg2_end], axis=1))[0, 1]
    melody_contour_dist = np.abs(np.mean(melody_contour[seg1_start:seg1_end]) - np.mean(melody_contour[seg2_start:seg2_end]))
    
    return (
        mfcc_dist < mfcc_threshold and
        chroma_corr > chroma_threshold and
        spectral_contrast_corr > spectral_contrast_threshold and
        tonnetz_corr > tonnetz_threshold and
        melody_contour_dist < melody_contour_threshold
    )

# 短いセグメントを統合, 採用した変化点も保存
use_change_points = []
filtered_change_samples = [change_samples[0]]
for i in range(1, len(change_samples)):
    prev_start = filtered_change_samples[-1] // hop_length
    prev_end = change_samples[i] // hop_length
    
    # 次のセグメントが短すぎるか、セグメントの結合条件を満たしているかを確認
    if (change_samples[i] - filtered_change_samples[-1] < min_segment_length or
        (i < len(change_samples) - 1 and
         should_merge_segments(
            prev_start,
            prev_end,
            change_samples[i] // hop_length,
            change_samples[i + 1] // hop_length,
            mfcc, chroma, spectral_contrast,
            tonnetz, melody_contour))):
        continue

    # セグメントを結合
    filtered_change_samples.append(change_samples[i])
    use_change_points.append(change_samples[i] // hop_length)

# 最後の変化点を追加
filtered_change_samples.append(change_samples[-1])

import os
filename = os.path.basename(filepath)
# 拡張子を削除
filename = os.path.splitext(filename)[0]
dirname = f'./output/{filename}'
if(os.path.exists(dirname) == True):
    # delete files
    for file in os.listdir(dirname):
        os.remove(os.path.join(dirname, file))
os.makedirs(dirname, exist_ok=True)

# セグメントごとにファイルに保存
for i in range(len(filtered_change_samples) - 1):
    start_sample = filtered_change_samples[i]
    end_sample = filtered_change_samples[i + 1]
    segment = y[start_sample:end_sample]
    
    output_filename = f'{dirname}/segment_{i+1}.wav'
    sf.write(output_filename, segment, sr)
    print(f'Saved {output_filename}')

# 結果をプロット
plt.figure(figsize=(14, 5))
plt.plot(combined_score, label='Combined Score')
plt.plot(dynamic_threshold, label='Dynamic Threshold', linestyle='dashed', color='red')
plt.plot(local_mean, label='Local Mean', linestyle='dashed', color='green')
plt.plot(lower_threshold, label='Lower Threshold', linestyle='dashed', color='orange')
plt.vlines(change_points, 0, max(combined_score), color='r', linestyle='dashed')
plt.vlines(use_change_points, 0, max(combined_score), color='g')
plt.legend()
plt.savefig(f'{dirname}/result.png')
plt.show()
