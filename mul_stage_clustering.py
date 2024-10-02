from clustering import clustering
from wav_separate import wav_separate
import wave
import sys
from calc_feat_value import calc_distribution
import numpy as np
import umap
# from sklearn.ensemble import IsolationForest
# LocalOutlierFactor
# from sklearn.neighbors import LocalOutlierFactor

def mul_stage_clustering(filePath, is_return_segement = False):
    segments = wav_separate(filePath)
    sr = wave.open(filePath, "rb").getframerate()
    feat_value_array = calc_distribution(segments, sr)

    reducer = umap.UMAP(n_components=2, random_state=42)

    # clf = LocalOutlierFactor(contamination='auto', novelty=False)
    # rms_outliers = clf.fit_predict(feat_value_array[:, [0, 1]])
    # f0_outliers = clf.fit_predict(feat_value_array[:, [2, 3]])
    # sb_outliers = clf.fit_predict(feat_value_array[:, [4, 5]])
    # zcr_outliers = clf.fit_predict(feat_value_array[:, [6, 7]])

    outlier_index = []

    # outlier_index = [np.where(rms_outliers == -1)[0], np.where(f0_outliers == -1)[0], np.where(sb_outliers == -1)[0], np.where(zcr_outliers == -1)[0]]
    # outlier_index = list(set(np.concatenate(outlier_index)))

    # 外れ値でないindexの配列を取得
    idx = [i for i in range(len(feat_value_array)) if i not in outlier_index]

    rms_feat_w_time = feat_value_array[:, [0, 1, 8]]
    f0_feat_w_time = feat_value_array[:, [2, 3, 8]]
    sb_feat_w_time = feat_value_array[:, [4, 5, 8]]
    zcr_feat_w_time = feat_value_array[:, [6, 7, 8]]

    # rms_feat_w_time = np.delete(rms_feat_w_time, outlier_index, 0)
    # f0_feat_w_time = np.delete(f0_feat_w_time, outlier_index, 0)
    # sb_feat_w_time = np.delete(sb_feat_w_time, outlier_index, 0)
    # zcr_feat_w_time = np.delete(zcr_feat_w_time, outlier_index, 0)

    rms_feat_w_time = reducer.fit_transform(rms_feat_w_time)
    f0_feat_w_time = reducer.fit_transform(f0_feat_w_time)
    sb_feat_w_time = reducer.fit_transform(sb_feat_w_time)
    zcr_feat_w_time = reducer.fit_transform(zcr_feat_w_time)
    
    rms_feat_w_time = (rms_feat_w_time - np.mean(rms_feat_w_time, axis=0)) / np.std(rms_feat_w_time, axis=0)
    f0_feat_w_time = (f0_feat_w_time - np.mean(f0_feat_w_time, axis=0)) / np.std(f0_feat_w_time, axis=0)
    sb_feat_w_time = (sb_feat_w_time - np.mean(sb_feat_w_time, axis=0)) / np.std(sb_feat_w_time, axis=0)
    zcr_feat_w_time = (zcr_feat_w_time - np.mean(zcr_feat_w_time, axis=0)) / np.std(zcr_feat_w_time, axis=0)

    ord = 2
    rms_l2 = np.linalg.norm(feat_value_array[:, [0, 1]], axis=1, ord=ord)
    f0_l2 = np.linalg.norm(feat_value_array[:, [2, 3]], axis=1, ord=ord)
    sb_l2 = np.linalg.norm(feat_value_array[:, [4, 5]], axis=1, ord=ord)
    zcr_l2 = np.linalg.norm(feat_value_array[:, [6, 7]], axis=1, ord=ord)

    rms_spr = np.max(rms_l2) - np.min(rms_l2)
    f0_spr = np.max(f0_l2) - np.min(f0_l2)
    sb_spr = np.max(sb_l2) - np.min(sb_l2)
    zcr_spr = np.max(zcr_l2) - np.min(zcr_l2)

    # sprの値が大きい順にfeatsに格納
    feats = [rms_feat_w_time, f0_feat_w_time, sb_feat_w_time, zcr_feat_w_time]
    sprs = [rms_spr, f0_spr, sb_spr, zcr_spr]
    feats = [x for _, x in sorted(zip(sprs, feats), reverse=True)]

    def unit_clustering(data, index, labels, i, stop_i = 4):
        if i == stop_i or len(index) < 2:
            return labels
        # label = 0, 1で再帰的にクラスタリング
        feat = data[i][index]
        feat = (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)
        label = (clustering(feat, "single", 2)["Single"] << i)
        for j, l in enumerate(label):
            labels[index[j]] = labels[index[j]] + l 
        
        data_index_0 = [k for k in range(len(label)) if label[k] == 0]
        data_index_1 = [k for k in range(len(label)) if label[k] == 1]
        return unit_clustering(data, [index[i] for i in data_index_0], labels, i + 1, stop_i) + unit_clustering(data, [index[i] for i in data_index_1], labels, i + 1, stop_i)
    
    labels = np.zeros(len(idx))
    stop_i = 2
    labels = unit_clustering(feats, [i for i in range(len(idx))], labels, 0, stop_i)
    labels = labels / np.gcd.reduce(labels.astype(int))

    label = np.zeros(len(feat_value_array))
    for i, l in enumerate(idx):
        label[l] = labels[i]
    for i in range(len(feat_value_array)):
        if i in outlier_index:
            label[i] = -1

    if is_return_segement:
        return [label, segments]
    return label

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mul_stage_clustering.py [input wav file]")
        sys.exit()
    filePath = sys.argv[1]
    labels, segments = mul_stage_clustering(filePath, True)
    print(labels.astype(int))
    # 隣り合ったセグメントのラベルが同じであれば結合する
    new_segments = []
    new_labels = []
    for i in range(len(labels)):
        if i == 0:
            new_segments.append(segments[i])
            new_labels.append(labels[i])
        elif labels[i] == labels[i - 1]:
            new_segments[-1] = np.concatenate([new_segments[-1], segments[i]])
        else:
            new_segments.append(segments[i])
            new_labels.append(labels[i])
    
    import librosa
    import wave
    import soundfile as sf
    import os
    dirname = "merged_segments"
    filename = os.path.basename(filePath)
    filename = os.path.splitext(filename)[0]
    output_dir = f'./{dirname}/{filename}'
    os.makedirs(output_dir, exist_ok=True)
    if(os.path.exists(output_dir) == True):
        # delete files
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    sr = wave.open(filePath, "rb").getframerate()
    for i, segment in enumerate(new_segments):
        sf.write(f"{output_dir}/_segment_{i}.wav", segment, sr)
    print("Output files are saved in " + output_dir)
