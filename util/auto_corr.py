def auto_corr(data):
    import numpy as np
    
    autocorr = np.correlate(data, data, mode='full')
    return autocorr

if __name__ == "__main__":
    import numpy as np
    import os
    import librosa
    import sys
    import matplotlib.pyplot as plt

    filePath = sys.argv[1]
    y, sr = librosa.load(filePath, sr = 4410)
    autocorr = auto_corr(y)
    plt.scatter(range(len(autocorr)), autocorr)

    plt.show()