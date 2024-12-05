import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
from spleeter.separator import Separator
import gc

def segment_separate_from_data(y, sr, filepath, saveFig=False):
    import re
    filename = re.split(r'[/\\]', filepath)[-1].split('.')[0]
    # save y in file
    savename = f'./tmp/{filename}.wav'
    sf.write(savename, y, sr)
    model = Separator('spleeter:2stems')
    model.separate_to_file(savename, f'./tmp/')
    del model
    y, sr = librosa.load(f'./tmp/{filename}/accompaniment.wav')
    
    # テンポとビートの検出
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # 1小節の長さを計算
    beats_per_measure = 4  # 4拍子として
    measure_length = int((tempo * sr) / 60 * beats_per_measure)

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
    melody_contour_diff = np.abs(np.diff(melody_contour))

    # 全ての特徴量の長さを一致させる
    min_length = min(len(energy_diff), len(zero_crossings_diff), len(spectral_flux_diff), len(melody_contour_diff))
    energy_diff = energy_diff[:min_length]
    zero_crossings_diff = zero_crossings_diff[:min_length]
    spectral_flux_diff = spectral_flux_diff[:min_length]
    melody_contour_diff = melody_contour_diff[:min_length]

    # 総合的なスコアを計算
    combined_score = (
        np.abs(energy_diff) +
        np.abs(zero_crossings_diff) +
        np.abs(spectral_flux_diff)
    )

    # Box-Cox変換
    from scipy.stats import boxcox
    combined_score = boxcox(combined_score + 1)[0]

    # 局所的な平均と標準偏差を計算
    def valid_convolve(xx, size):
        import math
        b = np.ones(size)/size
        xx_mean = np.convolve(xx, b, mode="same")

        n_conv = math.ceil(size/2)

        # 補正部分
        xx_mean[0] *= size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= size/(i+n_conv)
            xx_mean[-i] *= size/(i + n_conv - (size % 2)) 
	    # size%2は奇数偶数での違いに対応するため

        return xx_mean
    window_size = int(len(combined_score) / (len(y) / sr) * (tempo / 60 * 8))  # 窓サイズ
    local_mean = valid_convolve(combined_score, window_size)
    local_std = np.sqrt(valid_convolve((combined_score - local_mean[:len(combined_score)])**2, window_size))

    # 動的な閾値を設定
    n = 2.
    dynamic_threshold  = local_mean + n * local_std

    # 変化点の検出
    # 前後のwindow_size分は変化点としない
    # combined_score[:window_size] = 0
    # combined_score[-window_size:] = 0
    change_points = np.where(combined_score > dynamic_threshold[:len(combined_score)])[0]

    # 変化点をサンプル単位に変換
    change_samples = change_points * hop_length

    # 境界点に先頭と末尾を追加
    change_samples = np.concatenate(([0], change_samples, [len(y)]))

    # 最小セグメント長をサンプル単位で設定
    min_segment_length = measure_length // 2

    # 短いセグメントを統合するロジック
    from util.matrix import cos_sim
    threshold = 0.98
    def should_merge_segments(seg1_start, seg1_end, seg2_start, seg2_end, mfcc, chroma, spectral_contrast, tonnetz, 
                              mfcc_threshold=threshold, chroma_threshold=threshold, spectral_contrast_threshold=threshold, 
                              tonnetz_threshold=threshold):

        mfcc_bin_seg1 = mfcc[:, seg1_start:seg1_end]
        mfcc_bin_seg2 = mfcc[:, seg2_start:seg2_end]
        chroma_bin_seg1 = chroma[:, seg1_start:seg1_end]
        chroma_bin_seg2 = chroma[:, seg2_start:seg2_end]
        spectral_contrast_bin_seg1 = spectral_contrast[:, seg1_start:seg1_end]
        spectral_contrast_bin_seg2 = spectral_contrast[:, seg2_start:seg2_end]
        tonnetz_bin_seg1 = tonnetz[:, seg1_start:seg1_end]
        tonnetz_bin_seg2 = tonnetz[:, seg2_start:seg2_end]

        mfcc_cos_sim = cos_sim(mfcc_bin_seg1, mfcc_bin_seg2)
        chroma_cos_sim = cos_sim(chroma_bin_seg1, chroma_bin_seg2)
        spectral_contrast_cos_sim = cos_sim(spectral_contrast_bin_seg1, spectral_contrast_bin_seg2)
        tonnetz_cos_sim = cos_sim(tonnetz_bin_seg1, tonnetz_bin_seg2)

        return (
            mfcc_cos_sim > mfcc_threshold and
            chroma_cos_sim > chroma_threshold and
            spectral_contrast_cos_sim > spectral_contrast_threshold and
            tonnetz_cos_sim > tonnetz_threshold
        )

    # 短いセグメントを統合, 採用した変化点も保存
    use_change_points = []
    filtered_change_samples = [change_samples[0]]
    for i in range(1, len(change_samples)):
        prev_start = filtered_change_samples[-1] // hop_length
        prev_end = change_samples[i] // hop_length

        if (change_samples[i] - filtered_change_samples[-1] < min_segment_length or
            (i < len(change_samples) - 1 and
             should_merge_segments(
                prev_start,
                prev_end,
                change_samples[i] // hop_length,
                change_samples[i + 1] // hop_length,
                mfcc, chroma, spectral_contrast,
                tonnetz))):
            continue
        
        filtered_change_samples.append(change_samples[i])
        use_change_points.append(change_samples[i] // hop_length)

    # 最後の変化点を追加
    filtered_change_samples.append(change_samples[-1])
    
    import os
    filename = os.path.basename(filepath)
    # 拡張子を削除
    filename = os.path.splitext(filename)[0]
    dirname = f'./output2/{filename}'
    if(os.path.exists(dirname) == True):
        # delete files
        for file in os.listdir(dirname):
            if(file.endswith('.wav')):
                os.remove(os.path.join(dirname, file))
    os.makedirs(dirname, exist_ok=True)
    
    if saveFig:
        # 結果をプロット
        import time
        plt.figure(figsize=(16, 9))
        plt.plot(combined_score, label='Combined Score')
        plt.plot(dynamic_threshold, label='Dynamic Threshold', linestyle='dashed', color='red')
        plt.vlines(use_change_points, 0, max(combined_score), color='brown', linewidth=3)
        plt.savefig(f'{dirname}/{time.time()}change_points.png', format='png', dpi=600)
        del time
        plt.close()

    if __name__ == '__main__':
        # セグメントごとにファイルに保存
        for i in range(len(filtered_change_samples) - 1):
            start_sample = filtered_change_samples[i]
            end_sample = filtered_change_samples[i + 1]
            segment = y[start_sample:end_sample]

            output_filename = f'{dirname}/segment_{i+1}.wav'
            sf.write(output_filename, segment, sr)
            print(f'Saved {output_filename}')

    gc.collect()
    return np.array(filtered_change_samples) / sr


def segment_separate(filepath, saveFig=False):
    y, sr = librosa.load(filepath)
    return segment_separate_from_data(y, sr, filepath, saveFig)    

if __name__ == '__main__':
    import sys
    import os
    import re
    try:
        input_file = sys.argv[1]
    except:
        print("Usage: python app.py <input_file>")
        sys.exit(1)
    # 分離モードを指定

    filename = re.split(r'[/\\]', input_file)[-1]
    print(filename)
    filename = filename.split(".")[0]
    print(filename)
    dir = f"./separate_data/{filename}"
    if(os.path.exists(dir) != True):
        from spleeter.separator import Separator
        separator = Separator("spleeter:2stems")
        # インプットファイルと出力ディレクトリを指定して分離実行
        separator.separate_to_file(input_file, "./separate_data/")

    # セグメント分離
    print(dir)
    separated_files_name = os.listdir(dir)
    print(separated_files_name)
    for file_name in separated_files_name:
        if file_name.endswith('.wav'):
            segment_separate(f'{dir}/{file_name}', saveFig=True)