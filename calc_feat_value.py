import librosa
import numpy as np
from chord.chord_estimation import chord_root_tone_estimate
import time

def calc(data_array, sr) -> np.ndarray:
    # data_arrayの中で最も短い長さを取得
    min_length = min([len(data) for data in data_array])

    # data_arrayの長さを最小長に揃える、長いデータは中央の部分を取得
    data_array = np.array([data[int((len(data) - min_length) / 2):int((len(data) + min_length) / 2)] for data in data_array])

    chroma_array = np.array([librosa.feature.chroma_stft(y=data, sr=sr) for data in data_array])
    mfcc_array = np.array([librosa.feature.mfcc(y=data, sr=sr, n_mfcc=48) for data in data_array])
    chord_array = np.array([chord_root_tone_estimate(chroma) for chroma in chroma_array])
    rms_array = np.array([librosa.feature.rms(y=data) for data in data_array])

    feat_value_array = []
    for i, data in enumerate(data_array):
        # normalize
        chroma = chroma_array[i]
        mfcc = mfcc_array[i]
        rms = rms_array[i]
        feat_value_array.append(np.concatenate([chroma.flatten(), mfcc.flatten(), chord_array[i]]))

    return np.array(feat_value_array)

def calc_distribution(data_array, sr, is_raw = False) -> np.ndarray:
    rms_array = [librosa.feature.rms(y=data) for data in data_array]
    spectral_bandwidth_array = [librosa.feature.spectral_bandwidth(y=data, sr=sr) for data in data_array]
    zcr_array = [librosa.feature.zero_crossing_rate(y=data) for data in data_array]

    length_feat = np.array([len(data) for data in data_array])
    time_feat = np.array([i / sr for i in length_feat])
    
    feat_value_array = []
    for i, data in enumerate(data_array):
        rms_mean = np.mean(rms_array[i])
        rms_std = np.std(rms_array[i])

        spectral_bandwidth_mean = np.mean(spectral_bandwidth_array[i])
        spectral_bandwidth_std = np.std(spectral_bandwidth_array[i])

        zcr_mean = np.mean(zcr_array[i])
        zcr_std = np.std(zcr_array[i])

        f0 = librosa.yin(data, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.mean(f0)
        f0_std = np.std(f0)

        feat_value_array.append(np.array([
            rms_mean, rms_std,
            f0_mean, f0_std,
            spectral_bandwidth_mean, spectral_bandwidth_std,
            zcr_mean, zcr_std,
            time_feat[i]
            ]))
        
    feat_value_array = np.array(feat_value_array)

    if is_raw:
        return feat_value_array
    
    for i in range(feat_value_array.shape[1]):
        feat_value_array[:, i] = (feat_value_array[:, i] - np.mean(feat_value_array[:, i])) / np.std(feat_value_array[:, i])

    return np.array(feat_value_array)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    from wav_separate import wav_separate
    import umap
    import wave

    filePath = sys.argv[1]
    y, sr = librosa.load(filePath)
    segments = wav_separate(filePath)
    with wave.open(filePath, 'r') as wf:
        sr = wf.getframerate()
    feat_value_array = calc_distribution(segments, sr)

    reducer = umap.UMAP(n_components=2, random_state=42)
    feat_value_array_umap_o_time = reducer.fit_transform(feat_value_array[:6])
    feat_value_array_umap = reducer.fit_transform(feat_value_array)

    annotaion = [f"{i}" for i in range(len(feat_value_array))]
    
    fig = plt.figure()
    ax_1 = fig.add_subplot(231)
    ax_1.scatter(feat_value_array[:, 0], feat_value_array[:, 1])
    ax_1.set_title('RMS')
    for i, txt in enumerate(annotaion):
        ax_1.annotate(txt, (feat_value_array[i, 0], feat_value_array[i, 1]))

    ax_2 = fig.add_subplot(232)
    ax_2.scatter(feat_value_array[:, 2], feat_value_array[:, 3])
    ax_2.set_title('F0')
    for i, txt in enumerate(annotaion):
        ax_2.annotate(txt, (feat_value_array[i, 2], feat_value_array[i, 3]))

    ax_3 = fig.add_subplot(233)
    ax_3.scatter(feat_value_array[:, 4], feat_value_array[:, 5])
    ax_3.set_title('Spectral Bandwidth')
    for i, txt in enumerate(annotaion):
        ax_3.annotate(txt, (feat_value_array[i, 4], feat_value_array[i, 5]))

    ax_4 = fig.add_subplot(234)
    ax_4.scatter(feat_value_array[:, 6], feat_value_array[:, 7])
    ax_4.set_title('ZCR')
    for i, txt in enumerate(annotaion):
        ax_4.annotate(txt, (feat_value_array[i, 6], feat_value_array[i, 7]))

    ax_5 = fig.add_subplot(235)
    ax_5.scatter(feat_value_array_umap_o_time[:, 0], feat_value_array_umap_o_time[:, 1])
    ax_5.set_title('UMAP without time')
    for i, txt in enumerate(annotaion[:6]):
        ax_5.annotate(txt, (feat_value_array_umap_o_time[i, 0], feat_value_array_umap_o_time[i, 1]))

    ax_6 = fig.add_subplot(236)
    ax_6.scatter(feat_value_array_umap[:, 0], feat_value_array_umap[:, 1])
    ax_6.set_title('UMAP')
    for i, txt in enumerate(annotaion):
        ax_6.annotate(txt, (feat_value_array_umap[i, 0], feat_value_array_umap[i, 1]))

    plt.tight_layout()

    plt.show()