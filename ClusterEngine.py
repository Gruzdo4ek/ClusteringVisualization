from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class KMeansClustering:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data_scaled = None

    def fit_predict(self, data, n_clusters=None, num_classes=None, visualize=True):
        # Масштабируем данные
        self.data_scaled = self.scaler.fit_transform(data)

        # Если количество кластеров не задано, но известно количество классов
        if n_clusters is None and num_classes is not None:
            n_clusters = num_classes

        # Если количество кластеров всё ещё не определено, находим оптимальное
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(self.data_scaled)

        # Обучение модели
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = self.model.fit_predict(self.data_scaled)

        return labels

    def find_optimal_clusters(self, data, max_k=10):
        best_k = 2
        best_score = -1
        scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k