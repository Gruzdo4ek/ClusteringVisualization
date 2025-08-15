import numpy as np
import cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from DC import local_search_scheme


# ==================== KMeans ==================== #
class KMeansClustering:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data_scaled = None

    def fit_predict(self, data, n_clusters=None, num_classes=None, visualize=True):
        self.data_scaled = self.scaler.fit_transform(data)

        if n_clusters is None and num_classes is not None:
            n_clusters = num_classes

        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(self.data_scaled)

        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.model.fit_predict(self.data_scaled)
        return labels

    def find_optimal_clusters(self, data, max_k=10):
        best_k = 2
        best_score = -1

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k


# ==================== DC-алгоритм (GSS) ==================== #

class DCClustering:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_predict(self, data, n_clusters=None, num_classes=None, visualize=True):
        data_scaled = self.scaler.fit_transform(data)

        if n_clusters is None and num_classes is not None:
            n_clusters = num_classes
        if n_clusters is None:
            n_clusters = 3  # можно добавить автоматический выбор

        x, y = local_search_scheme(data_scaled, k=n_clusters, d1=0.5, d2=0.5, tau=5)
        labels = np.argmax(x, axis=1)
        return labels



