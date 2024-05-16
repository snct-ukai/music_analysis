import numpy as np

def normalize_feature(feature, axis = None):
    return (feature - np.mean(feature, axis=axis)) / np.std(feature, axis=axis)