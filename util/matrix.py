import numpy as np

def binarize_feature_matrix(feature_matrix, threshold):
    return (feature_matrix > threshold).astype(int)

def jaccard_similarity_matrix(matrix1, matrix2):
    return len(np.intersect1d(matrix1,matrix2))/len(np.union1d(matrix1,matrix2))

def cos_sim(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    # matrixの列ベクトルの単位ベクトル化
    matrix1 = matrix1 / np.linalg.norm(matrix1, axis=0)
    matrix2 = matrix2 / np.linalg.norm(matrix2, axis=0)
    matrix_product = np.dot(matrix1.T, matrix2)
    cos_sim = np.max(matrix_product)
    return cos_sim