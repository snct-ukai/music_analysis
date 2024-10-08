import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def kmeans(data : np.ndarray, N = 7) -> np.ndarray:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=N).fit(data)
    labels = kmeans.labels_
    return labels

# ウォード法
def ward(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='ward')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

# 群平均法
def average(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='average')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

# 重心法
def centroid(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='centroid')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

# メディアン法
def median(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='median')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

# 単純連結法
def single(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='single')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

# 完全連結法
def complete(data : np.ndarray, N = 7, ax = None) -> np.ndarray:
    z = linkage(data, method='complete')
    labels = fcluster(z, N, criterion='maxclust') - 1
    if ax is not None:
        dendrogram(z, ax=ax)
    return labels

def xmeans(data : np.ndarray) -> np.ndarray:
    from pyclustering.cluster.xmeans import xmeans
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

    initial_centers = kmeans_plusplus_initializer(data, 2).initialize()
    xmeans_instance = xmeans(data, initial_centers, kmax=10)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    labels = np.zeros(data.shape[0])
    for i, cluster in enumerate(clusters):
        labels[cluster] = i
    return labels

def hdbscan(data : np.ndarray) -> np.ndarray:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    clusterer.fit(data)
    return clusterer.labels_

def gmm(data : np.ndarray, N = 7) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=N).fit(data)
    labels = gmm.predict(data)
    return labels

def kullback_leibler_divergence(data: np.ndarray) -> np.ndarray:
    # 多次元ベクトルのKullback-Leiblerダイバージェンスを計算
    def KLD(x, y):
      """Compute the Kullback-Leibler divergence between two multivariate samples.
      Parameters
      ----------
      x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
      y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.
      Returns
      -------
      out : float
        The estimated Kullback-Leibler divergence D(P||Q).
      References
      ----------
      Pérez-Cruz, F. Kullback-Leibler divergence estimation of
    continuous distributions IEEE International Symposium on Information
    Theory, 2008.
      """
      from scipy.spatial import cKDTree as KDTree

      # Check the dimensions are consistent
      x = np.atleast_2d(x)
      y = np.atleast_2d(y)

      n,d = x.shape
      m,dy = y.shape

      assert(d == dy)


      # Build a KD tree representation of the samples and find the nearest neighbour
      # of each point in x.
      xtree = KDTree(x)
      ytree = KDTree(y)

      # Get the first two nearest neighbours for x, since the closest one is the
      # sample itself.
      r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
      s = ytree.query(x, k=1, eps=.01, p=2)[0]

      # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
      # on the first term of the right hand side.
      return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

    kl_divergence = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            kl_divergence[i, j] = KLD(data[i], data[j])

    labels = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        labels[i] = np.argmin(kl_divergence[i])
    return labels

def k_shape(data: np.ndarray, N = 7) -> np.ndarray:
    from tslearn.clustering import KShape
    ks = KShape(n_clusters=N)
    labels = ks.fit_predict(data)
    return labels

if __name__ == "__main__":
    import sys
    import wave
    from wav_separate import wav_separate
    import calc_feat_value
    import matplotlib.pyplot as plt

    filePath = sys.argv[1]
    segments = wav_separate(filePath)
    sr = wave.open(filePath, "rb").getframerate()
    feat_value_array = calc_feat_value.calc_distribution(segments, sr)
    #  use f0 and time
    feat_value_array = feat_value_array[:, [2, 3, 8]]

    fig = plt.figure()
    ax_1 = fig.add_subplot(221)
    ax_2 = fig.add_subplot(222)
    ax_3 = fig.add_subplot(223)
    ax_4 = fig.add_subplot(224)

    ward(feat_value_array, 2, ax=ax_1)
    ax_1.set_title('Ward')
    average(feat_value_array, 2, ax=ax_2)
    ax_2.set_title('Average')
    centroid(feat_value_array, 2, ax=ax_3)
    ax_3.set_title('Centroid')
    median(feat_value_array, 2, ax=ax_4)
    ax_4.set_title('Median')
    plt.show()