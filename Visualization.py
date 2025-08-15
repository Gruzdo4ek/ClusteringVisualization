from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d


class Visualizer:
    def _draw_cluster_plot(self, ax, data, labels, title="Результаты кластеризации"):
        # Применяем PCA, если данных больше 2 признаков
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)

        centroids = np.array([data[labels == label].mean(axis=0)
                              for label in np.unique(labels)])

        # Диаграмма Вороного
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                        line_colors='orange', line_width=2, line_alpha=0.6)

        # Точки данных
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels,
                             cmap='viridis', s=20, alpha=0.7)

        # Центроиды
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='X', s=50, linewidths=2)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Компонента 1", fontsize=8)
        ax.set_ylabel("Компонента 2", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(*scatter.legend_elements(), title="Кластеры", loc='upper right',fontsize=6, title_fontsize=8)

    def plot_reference_clusters(self, data, reference_labels):
        """Отрисовывает эталонные кластеры (если они есть)"""
        if reference_labels is not None:
            self._draw_cluster_plot(data, reference_labels, "Эталонные кластеры")

    def get_matplotlib_widget(self, data, labels, title="Кластеры"):
        # Создаем Figure и Canvas
        fig = Figure(figsize=(3, 3))
        ax = fig.add_subplot(111)

        # Рисуем график на оси
        self._draw_cluster_plot(ax, data, labels, title)

        # Canvas для вставки в PySide6
        canvas = FigureCanvas(fig)

        # Контейнер с layout для масштабирования
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        return container