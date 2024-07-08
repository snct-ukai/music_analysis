from wav_separate import wav_separate
import numpy as np
from segment_clustering import kmeans, ward, average, centroid, median, xmeans
import calc_feat_value
import wave
import pickle
import matplotlib.pyplot as plt

N = 2

def segment_clustering(filePath):
    segments = wav_separate(filePath)
    sr = wave.open(filePath, "rb").getframerate()
    feat_value_array = calc_feat_value.calc_distribution(segments, sr)[:,2:4]
    clustering(feat_value_array, mode = "all", is_plt = True)

def clustering(feat_value_array, mode = "ward", is_plt = False):
    annotaion = [f"{i}" for i in range(len(feat_value_array))]
    
    titles = []
    clustering_methods = []

    if mode == "all":
        mode = "kmeans,ward,average,centroid,median"

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
    
    if is_plt:
        fig = plt.figure()
        color_array = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, clustering_method in enumerate(clustering_methods):
            result = clustering_method(feat_value_array, N)
            ax = fig.add_subplot(2, int((2 * (len(titles) / 2) + 1) / 2), 1 + i)
            # kmeans の結果から色で分けてプロット
            for j in range(feat_value_array.shape[0]):
                ax.scatter(feat_value_array[j, 0], feat_value_array[j, 1], color=color_array[result[j]])
            ax.set_title(titles[i])
            for j, txt in enumerate(annotaion):
                ax.annotate(txt, (feat_value_array[j, 0], feat_value_array[j, 1]))
        plt.show()

if __name__ == "__main__":
    import sys
    filePath = sys.argv[1]
    segment_clustering(filePath)