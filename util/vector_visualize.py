from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

tsne = TSNE(n_components=2, random_state=42)
def vector_visualize(vectors : np.ndarray) -> np.ndarray:
    return tsne.fit_transform(vectors)
