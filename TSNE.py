import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from util import vector_visualize
import librosa

mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

audioFileDir = "./wav"
# get audio file paths
audioFiles = librosa.util.find_files(audioFileDir)
# load audio files
audioData = [librosa.load(audioFile, sr=3600) for audioFile in audioFiles]

y, sr = audioData[0]

spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spectral_bandwidth /= np.max(spectral_bandwidth)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_centroid /= np.max(spectral_centroid)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
spectral_rolloff /= np.max(spectral_rolloff)
rms = librosa.feature.rms(y=y)[0]
rms /= np.max(rms)
zcr = librosa.feature.zero_crossing_rate(y=y)[0]
zcr /= np.max(zcr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

mfcc_1 = mfcc[0]
mfcc_1 /= np.max(mfcc_1)
mfcc_2 = mfcc[1]
mfcc_2 /= np.max(mfcc_2)
mfcc_3 = mfcc[2]
mfcc_3 /= np.max(mfcc_3)
mfcc_4 = mfcc[3]
mfcc_4 /= np.max(mfcc_4)
mfcc_5 = mfcc[4]
mfcc_5 /= np.max(mfcc_5)
mfcc_6 = mfcc[5]
mfcc_6 /= np.max(mfcc_6)
mfcc_7 = mfcc[6]
mfcc_7 /= np.max(mfcc_7)
mfcc_8 = mfcc[7]
mfcc_8 /= np.max(mfcc_8)
mfcc_9 = mfcc[8]
mfcc_9 /= np.max(mfcc_9)
mfcc_10 = mfcc[9]
mfcc_10 /= np.max(mfcc_10)
mfcc_11 = mfcc[10]
mfcc_11 /= np.max(mfcc_11)
mfcc_12 = mfcc[11]
mfcc_12 /= np.max(mfcc_12)

chroma = librosa.feature.chroma_stft(y=y, sr=sr)

chroma_1 = chroma[0]
chroma_1 /= np.max(chroma_1)
chroma_2 = chroma[1]
chroma_2 /= np.max(chroma_2)
chroma_3 = chroma[2]
chroma_3 /= np.max(chroma_3)
chroma_4 = chroma[3]
chroma_4 /= np.max(chroma_4)
chroma_5 = chroma[4]
chroma_5 /= np.max(chroma_5)
chroma_6 = chroma[5]
chroma_6 /= np.max(chroma_6)
chroma_7 = chroma[6]
chroma_7 /= np.max(chroma_7)
chroma_8 = chroma[7]
chroma_8 /= np.max(chroma_8)
chroma_9 = chroma[8]
chroma_9 /= np.max(chroma_9)
chroma_10 = chroma[9]
chroma_10 /= np.max(chroma_10)
chroma_11 = chroma[10]
chroma_11 /= np.max(chroma_11)
chroma_12 = chroma[11]
chroma_12 /= np.max(chroma_12)

playtime = y.size / sr
div_num = 25
div_range = 1 / div_num

l_spectral_bandwidth = spectral_bandwidth.size
l_spectral_centroid = spectral_centroid.size
l_spectral_rolloff = spectral_rolloff.size
l_rms = rms.size
l_zcr = zcr.size
l_mfcc_1 = mfcc_1.size
l_mfcc_2 = mfcc_2.size
l_mfcc_3 = mfcc_3.size
l_mfcc_4 = mfcc_4.size
l_mfcc_5 = mfcc_5.size
l_mfcc_6 = mfcc_6.size
l_mfcc_7 = mfcc_7.size
l_mfcc_8 = mfcc_8.size
l_mfcc_9 = mfcc_9.size
l_mfcc_10 = mfcc_10.size
l_mfcc_11 = mfcc_11.size
l_mfcc_12 = mfcc_12.size
l_chroma_1 = chroma_1.size
l_chroma_2 = chroma_2.size
l_chroma_3 = chroma_3.size
l_chroma_4 = chroma_4.size
l_chroma_5 = chroma_5.size
l_chroma_6 = chroma_6.size
l_chroma_7 = chroma_7.size
l_chroma_8 = chroma_8.size
l_chroma_9 = chroma_9.size
l_chroma_10 = chroma_10.size
l_chroma_11 = chroma_11.size
l_chroma_12 = chroma_12.size

from sklearn.manifold import TSNE

tsne_engine = TSNE(n_components=2, random_state=42)
d = []
for i in range(5):
    for j in range(5):
        data = np.array([
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range - 3)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range - 3)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range - 3)],
            rms[int((5*i+j) * l_rms * div_range - 3)],
            zcr[int((5*i+j) * l_zcr * div_range - 3)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range - 3)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range - 3)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range - 3)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range - 3)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range - 3)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range - 3)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range - 3)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range - 3)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range - 3)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range - 3)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range - 3)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range - 3)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range - 2)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range - 2)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range - 2)],
            rms[int((5*i+j) * l_rms * div_range - 2)],
            zcr[int((5*i+j) * l_zcr * div_range - 2)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range - 2)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range - 2)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range - 2)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range - 2)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range - 2)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range - 2)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range - 2)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range - 2)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range - 2)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range - 2)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range - 2)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range - 2)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range - 1)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range - 1)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range - 1)],
            rms[int((5*i+j) * l_rms * div_range - 1)],
            zcr[int((5*i+j) * l_zcr * div_range - 1)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range - 1)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range - 1)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range - 1)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range - 1)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range - 1)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range - 1)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range - 1)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range - 1)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range - 1)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range - 1)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range - 1)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range - 1)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range)],
            rms[int((5*i+j) * l_rms * div_range)],
            zcr[int((5*i+j) * l_zcr * div_range)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 1)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 1)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 1)],
            rms[int((5*i+j) * l_rms * div_range + 1)],
            zcr[int((5*i+j) * l_zcr * div_range + 1)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 1)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 1)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 1)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 1)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 1)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 1)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 1)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 1)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 1)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 1)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 1)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 1)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 2)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 2)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 2)],
            rms[int((5*i+j) * l_rms * div_range + 2)],
            zcr[int((5*i+j) * l_zcr * div_range + 2)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 2)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 2)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 2)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 2)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 2)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 2)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 2)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 2)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 2)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 2)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 2)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 2)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 3)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 3)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 3)],
            rms[int((5*i+j) * l_rms * div_range + 3)],
            zcr[int((5*i+j) * l_zcr * div_range + 3)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 3)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 3)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 3)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 3)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 3)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 3)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 3)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 3)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 3)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 3)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 3)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 3)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 4)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 4)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 4)],
            rms[int((5*i+j) * l_rms * div_range + 4)],
            zcr[int((5*i+j) * l_zcr * div_range + 4)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 4)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 4)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 4)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 4)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 4)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 4)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 4)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 4)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 4)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 4)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 4)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 4)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 5)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 5)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 5)],
            rms[int((5*i+j) * l_rms * div_range + 5)],
            zcr[int((5*i+j) * l_zcr * div_range + 5)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 5)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 5)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 5)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 5)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 5)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 5)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 5)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 5)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 5)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 5)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 5)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 5)],
            spectral_bandwidth[int((5*i+j) * l_spectral_bandwidth * div_range + 6)],
            spectral_centroid[int((5*i+j) * l_spectral_centroid * div_range + 6)],
            spectral_rolloff[int((5*i+j) * l_spectral_rolloff * div_range + 6)],
            rms[int((5*i+j) * l_rms * div_range + 6)],
            zcr[int((5*i+j) * l_zcr * div_range + 6)],
            mfcc_1[int((5*i+j) * l_mfcc_1 * div_range + 6)],
            mfcc_2[int((5*i+j) * l_mfcc_2 * div_range + 6)],
            mfcc_3[int((5*i+j) * l_mfcc_3 * div_range + 6)],
            mfcc_4[int((5*i+j) * l_mfcc_4 * div_range + 6)],
            mfcc_5[int((5*i+j) * l_mfcc_5 * div_range + 6)],
            mfcc_6[int((5*i+j) * l_mfcc_6 * div_range + 6)],
            mfcc_7[int((5*i+j) * l_mfcc_7 * div_range + 6)],
            mfcc_8[int((5*i+j) * l_mfcc_8 * div_range + 6)],
            mfcc_9[int((5*i+j) * l_mfcc_9 * div_range + 6)],
            mfcc_10[int((5*i+j) * l_mfcc_10 * div_range + 6)],
            mfcc_11[int((5*i+j) * l_mfcc_11 * div_range + 6)],
            mfcc_12[int((5*i+j) * l_mfcc_12 * div_range + 6)],
        ])
        d.append(data)
        print(data.shape)
d = np.array(d)
print(d.shape)
tsne = tsne_engine.fit_transform(d)
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.show()