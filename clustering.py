from wav_separate import wav_separate
import numpy as np
from segment_clustering import kmeans, ward, average, centroid, median, xmeans, k_shape
import calc_feat_value
import wave
import pickle
import matplotlib.pyplot as plt

def segment_clustering(filePath):
    segments = wav_separate(filePath)
    sr = wave.open(filePath, "rb").getframerate()
    feat_value_array = calc_feat_value.calc_distribution(segments, sr)
    
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)

    rms_feat_w_time = reducer.fit_transform(feat_value_array[:, [0, 1, 8]])
    f0_feat_w_time = reducer.fit_transform(feat_value_array[:, [2, 3, 8]])
    sb_feat_w_time = reducer.fit_transform(feat_value_array[:, [4, 5, 8]])
    zcr_feat_w_time = reducer.fit_transform(feat_value_array[:, [6, 7, 8]])

    feats = [rms_feat_w_time, f0_feat_w_time, sb_feat_w_time, zcr_feat_w_time]

def clustering(feat_value_array, mode = "ward", N = 2, is_plt = False):
    annotaion = [f"{i}" for i in range(len(feat_value_array))]
    
    titles = []
    clustering_methods = []
    results = dict()

    if mode == "all":
        mode = "kmeans,ward,average,centroid,median,k_shape"

    mode = mode.split(",")

    if mode.count("kmeans") > 0:
        clustering_methods.append(kmeans)
        titles.append("KMeans")
    if mode.count("ward") > 0:
        clustering_methods.append(ward)
        titles.append("Ward")
    if mode.count("average") > 0:
        clustering_methods.append(average)
        titles.append("Average")
    if mode.count("centroid") > 0:
        clustering_methods.append(centroid)
        titles.append("Centroid")
    if mode.count("median") > 0:
        clustering_methods.append(median)
        titles.append("Median")
    if mode.count("k_shape") > 0:
        clustering_methods.append(k_shape)
        titles.append("KShape")
    
    if is_plt:
        fig = plt.figure()
        import matplotlib
        color_array = matplotlib.colormaps["tab20"].colors
    for i, clustering_method in enumerate(clustering_methods):
        result = clustering_method(feat_value_array, N)
        results[titles[i]] = result
        if is_plt:
            ax = fig.add_subplot(2, int((2 * (len(titles) / 2) + 1) / 2), 1 + i)
            # kmeans の結果から色で分けてプロット
            for j in range(feat_value_array.shape[0]):
                ax.scatter(feat_value_array[j, 0], feat_value_array[j, 1], color=color_array[result[j]])
            ax.set_title(titles[i])
            for j, txt in enumerate(annotaion):
                ax.annotate(txt, (feat_value_array[j, 0], feat_value_array[j, 1]))
    if is_plt:
        plt.show()
    
    return results

if __name__ == "__main__":
    import sys
    filePath = sys.argv[1]
    segment_clustering(filePath)