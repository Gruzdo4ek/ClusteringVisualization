from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, v_measure_score
import numpy as np

class MetricsCalculator:
    def calculate_all(self, data, labels):
        metrics = {
            "Инерция": self.calculate_inertia(data, labels),
            "Индекс силуэта": silhouette_score(data, labels),
            "Индекс Давида-Болдуина": davies_bouldin_score(data, labels),
        }
        return metrics

    def calculate_comparative(self, true_labels, pred_labels):
        return {
            "Adjusted Rand Index": adjusted_rand_score(true_labels, pred_labels),
            "V-measure": v_measure_score(true_labels, pred_labels)
        }

    def calculate_inertia(self, data, labels):
        centers = {}
        for label in np.unique(labels):
            centers[label] = data[labels == label].mean(axis=0)

        inertia = 0
        for i, point in enumerate(data):
            center = centers[labels[i]]
            inertia += np.sum((point - center) ** 2)
        return inertia