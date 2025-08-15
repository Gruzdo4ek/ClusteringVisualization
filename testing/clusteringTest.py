
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pair_confusion_matrix, silhouette_score, davies_bouldin_score
import pandas as pd
import argparse
import time

from DC import local_search_scheme

np.random.seed(0)

# -----------------------
# Генерация данных
# -----------------------
def generate_dataset(m, k, typ='I', dim=2):
    grid_size = int(np.ceil(np.sqrt(k)))
    coords = []
    spacing = 6.0 if typ == 'I' else 4.0
    for i in range(grid_size):
        for j in range(grid_size):
            coords.append(np.array([i*spacing, j*spacing], dtype=float))
    coords = np.array(coords[:k])
    coords += np.random.randn(*coords.shape) * (0.3 if typ == 'I' else 0.6)
    counts = np.full(k, m // k)
    counts[: (m % k)] += 1
    points = []
    labels = []
    for i in range(k):
        std = 0.6 if typ == 'I' else 1.6
        pts = coords[i] + np.random.randn(counts[i], dim) * std
        points.append(pts)
        labels.extend([i] * counts[i])
    data = np.vstack(points)
    labels = np.array(labels)
    perm = np.random.permutation(m)
    return data[perm], labels[perm], coords

def objective_value(data, centers):
    dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2) ** 2
    return np.sum(np.min(dists, axis=1))

def pairwise_metrics(true_labels, pred_labels):
    pm = pair_confusion_matrix(true_labels, pred_labels)
    TN, FP, FN, TP = pm.ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f

def calculate_inertia(data, labels):
    inertia = 0.0
    for cluster_id in np.unique(labels):
        cluster_points = data[labels == cluster_id]
        center = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - center) ** 2)
    return inertia


def run_experiment(m, k, typ, trials=5, show_time=False):
    data, true_labels, true_centers = generate_dataset(m, k, typ=typ)

    # Единая стартовая точка
    np.random.seed(42)  # фиксируем для воспроизводимости
    x_init = np.random.rand(m, k)
    x_init = x_init / x_init.sum(axis=1, keepdims=True)
    idx = np.random.choice(m, k, replace=False)
    y_init = data[idx].copy()

    # --- k-means ---
    t_km = time.time()
    km = KMeans(n_clusters=k, init=y_init, n_init=1, max_iter=300, random_state=42)
    km.fit(data)
    t1 = time.time() - t_km
    obj_km = objective_value(data, km.cluster_centers_)
    p_km, r_km, f_km = pairwise_metrics(true_labels, km.labels_)

    metrics_km = {
        "Objective": obj_km,
        "Precision": p_km,
        "Recall": r_km,
        "F-measure": f_km,
        "Inertia": calculate_inertia(data, km.labels_),
        "Silhouette": silhouette_score(data, km.labels_),
        "Davies-Bouldin": davies_bouldin_score(data, km.labels_),
        "time": t1
    }

    # --- LSA ---
    t_lsa = time.time()
    x_lsa, y_lsa = local_search_scheme(data, k, d1=0.1, d2=0.55, x0=x_init, y0=y_init)
    t2 = time.time() - t_lsa
    labels_lsa = np.argmin(np.sum((data[:, None, :] - y_lsa[None, :, :]) ** 2, axis=2), axis=1)
    obj_lsa = objective_value(data, y_lsa)
    p_lsa, r_lsa, f_lsa = pairwise_metrics(true_labels, labels_lsa)

    metrics_lsa = {
        "Objective": obj_lsa,
        "Precision": p_lsa,
        "Recall": r_lsa,
        "F-measure": f_lsa,
        "Inertia": calculate_inertia(data, labels_lsa),
        "Silhouette": silhouette_score(data, labels_lsa),
        "Davies-Bouldin": davies_bouldin_score(data, labels_lsa),
        "time": t2
    }

    results = [
        ("k-means", metrics_km),
        ("LSA", metrics_lsa)
    ]
    return data, true_labels, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run smaller quick tests')
    args = parser.parse_args()

    if args.fast:
        tests = [(200, 16), (300, 16)]
    else:
        tests =  [(200, 16), (400, 16), (500, 25), (600, 25)]

    all_rows = []
    for typ in ['I', 'III']:
        for m, k in tests:
            print(f"Running: Type {typ}, m={m}, k={k}")
            data, true_labels, results = run_experiment(m, k, typ, trials=5, show_time=True)
            for alg, metrics in results:
                all_rows.append({
                    'Type': typ, 'm': m, 'k': k, 'Algorithm': alg,
                    **metrics
                })
    df = pd.DataFrame(all_rows)
    # Печать результатов в читаемом виде
    print("\n=== Results ===")
    print(df.to_string())

if __name__ == "__main__":
    main()
