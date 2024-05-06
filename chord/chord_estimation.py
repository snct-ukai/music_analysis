import numpy as np
import chord_vector as cv

def chord_estimate(chroma : np.ndarray, tempo : int, playtime : int):
    CPB = int(60 * len(chroma[0]) / (tempo * playtime))
    # sum chroma vectors for each measure
    chroma_sum = np.array([np.sum(chroma[:, i:i + CPB], axis=1) for i in range(0, len(chroma[0]), CPB)])

    chord_matching_score = np.dot(cv.templates, chroma_sum.T)
    chord_estimation = np.argmax(chord_matching_score, axis=0)
    chord_name = [cv.chord_dic[chord] for chord in chord_estimation]
    return chord_name