from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


class Visualizer:
    def plot_clusters(self, data, labels, title="Результаты кластеризации"):

        # Применяем PCA, если данных больше 2 признаков
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)

        centroids = np.array([data[labels == label].mean(axis=0)
                              for label in np.unique(labels)])

        # Диаграмма Вороного
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, show_points=False, show_vertices=False,
                        line_colors='orange', line_width=2, line_alpha=0.6)

        # Точки данных
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels,
                              cmap='viridis', s=50, alpha=0.7)

        # Центроиды
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=100, linewidths=2)

        plt.title(title, fontsize=14)
        plt.xlabel("Компонента 1", fontsize=12)
        plt.ylabel("Компонента 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.legend(*scatter.legend_elements(),
                   title="Кластеры",
                   loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_reference_clusters(self, data, reference_labels):
        """Отрисовывает эталонные кластеры (если они есть)"""
        if reference_labels is not None:
            self.plot_clusters(data, reference_labels, "Эталонные кластеры")