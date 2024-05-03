import numpy as np

def low_pass_filter(data, cutoff):
    N = len(data)
    F = np.fft.fft(data)
    G = F.copy()
    G[cutoff:N-cutoff] = 0
    return np.fft.ifft(G).real