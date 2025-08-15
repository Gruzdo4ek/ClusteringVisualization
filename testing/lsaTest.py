import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi, voronoi_plot_2d
from DC import local_search_scheme  # импортируешь свой алгоритм


class Visualizer:
    def plot_clusters(self, data, labels, title="Результаты кластеризации"):
        # Определяем размер фигуры на основе диапазона данных
        x_range = data[:, 0].max() - data[:, 0].min()
        y_range = data[:, 1].max() - data[:, 1].min()
        scale_factor = max(x_range, y_range) * 0.2  # 20% от максимального диапазона

        # Применяем PCA, если данных больше 2 признаков
        if data.shape[1] > 2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(data)

        centroids = np.array([data[labels == label].mean(axis=0)
                              for label in np.unique(labels)])

        # Рассчитываем границы для увеличения масштаба
        x_min, x_max = data[:, 0].min() - scale_factor, data[:, 0].max() + scale_factor
        y_min, y_max = data[:, 1].min() - scale_factor, data[:, 1].max() + scale_factor

        # Диаграмма Вороного
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, show_points=False, show_vertices=False,
                        line_colors='orange', line_width=2, line_alpha=0.6)

        # Точки данных с увеличенным размером маркеров
        scatter = plt.scatter(data[:, 0], data[:, 1], c=labels,
                              cmap='viridis', s=7, alpha=0.8, edgecolors='w', linewidth=0.5)

        # Центроиды с увеличенным размером
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c='red', marker='X', s=70, linewidths=2, zorder=10)

        # Настраиваем границы для лучшего масштабирования
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.title(title, fontsize=16)
        plt.xlabel("Компонента 1", fontsize=14)
        plt.ylabel("Компонента 2", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.legend(*scatter.legend_elements(),
                   title="Кластеры",
                   loc='best', fontsize=10)

        plt.tight_layout()
        plt.show()


def test_tau_effect():
    np.random.seed(42)
    data = np.random.rand(200, 2)
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    k = 8
    tau_values = [None, 5, 10, 15, 20, 25, 30]
    d1, d2 = 0.5, 0.5

    results = []
    visualizer = Visualizer()

    for tau in tau_values:
        start = time.time()
        x_opt, y_opt = local_search_scheme(data, k, d1, d2, tau=tau)
        labels = np.argmax(x_opt, axis=1)
        elapsed = time.time() - start

        sil = silhouette_score(data, labels)
        dbi = davies_bouldin_score(data, labels)
        var_size = np.var(np.sum(x_opt > 0.5, axis=0))

        tau_label = "Без tau" if tau is None else f"tau={tau}"
        results.append((tau_label, sil, dbi, var_size, elapsed))

        # Визуализация кластеров для текущего tau
        visualizer.plot_clusters(data, labels,
                                 title=f"Кластеризация (k=8, {tau_label})")

        # Вывод информации о размерах кластеров
        cluster_sizes = np.sum(x_opt > 0.5, axis=0)
        print(f"\nРаспределение по кластерам ({tau_label}):")
        for i, size in enumerate(cluster_sizes):
            print(f"Кластер {i + 1}: {size} точек")
        print(f"Мин. размер: {min(cluster_sizes)}, Макс. размер: {max(cluster_sizes)}")

    df = pd.DataFrame(results, columns=["tau", "Silhouette", "Davies-Bouldin", "Var Cluster Size", "Time (s)"])
    print("\nСводная таблица метрик:")
    print(df)

    # Графики метрик
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].plot(df["tau"], df["Silhouette"], marker="o")
    axs[0].set_title("Silhouette vs tau")
    axs[0].set_xlabel("tau")
    axs[0].set_ylabel("Silhouette")
    axs[0].grid(True)

    axs[1].plot(df["tau"], df["Davies-Bouldin"], marker="o", color="red")
    axs[1].set_title("Davies-Bouldin vs tau")
    axs[1].set_xlabel("tau")
    axs[1].set_ylabel("Davies-Bouldin")
    axs[1].grid(True)

    axs[2].plot(df["tau"], df["Var Cluster Size"], marker="o", color="green")
    axs[2].set_title("Дисперсия размеров кластеров vs tau")
    axs[2].set_xlabel("tau")
    axs[2].set_ylabel("Variance")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return df


if __name__ == "__main__":
    test_tau_effect()