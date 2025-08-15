# Импорт необходимых библиотек
import itertools

import numpy as np  # Для работы с массивами и математическими операциями
import cvxpy as cp  # Для решения выпуклых оптимизационных задач
import ecos  # Солвер ECOS для решения задач, используемый через cvxpy
import pandas as pd  # Для сохранения данных и результатов в CSV
from numpy.f2py.crackfortran import verbose
import time

from sklearn.metrics import silhouette_score


# Функция для вычисления функции g(x, y) — части целевой функции в DC-разложении
def compute_g(x, y, data, d1, d2):
    """Вычисляет значение функции g(x, y)"""
    k = y.shape[0]  # Получаем число кластеров k из размера массива центров y
    dists = cp.vstack([cp.norm(y[i] - data, axis=1) ** 2 for i in range(k)]).T  # Вычисляем квадраты евклидовых расстояний от каждой точки до каждого центра кластера
    return cp.sum(d1 * dists + d2 * cp.square(x)) # Вычисляем g(x, y) = d1 * sum(расстояния^2) + d2 * sum(x^2)


# Функция для вычисления функции h(x, y) — второй части DC-разложения
def compute_h(x, y, data, d1, d2):
    """Вычисляет значение функции h(x, y) """
    k = y.shape[0]
    dists = cp.vstack([cp.norm(y[i] - data, axis=1) ** 2 for i in range(k)]).T  # (m, k)
    return cp.sum(d1 * dists + d2 * cp.square(x) - cp.multiply(x, dists))


def compute_grad_h(x, y, data, d1, d2):
    """Оптимизированная версия: Вычисляет градиенты функции h(x, y) по x и y с векторизацией."""
    m, n = data.shape
    k = y.shape[0]

    # Векторизованное вычисление всех разностей: (k, m, n)
    dist_vec = y[:, np.newaxis, :] - data[np.newaxis, :, :]
    grad_y = np.sum(2 * (d1 - x.T[:, :, np.newaxis]) * dist_vec, axis=1)

    # Квадраты расстояний: sum по n (axis=2) → (k, m)
    dists_sq = np.sum(dist_vec ** 2, axis=2)
    grad_x = 2 * d2 * x - dists_sq.T

    return grad_x, grad_y

# Локальный поиск с использованием DC Algorithm (DCA)
def dca_local_search(initial_x, initial_y, data, d1, d2, tau=None, tol=1e-5, max_iter=30):
    # Копии начальных присваиваний x и центров y
    x = initial_x.copy()
    y = initial_y.copy()

    # Цикл до max_iter итераций
    for _ in range(max_iter):
        # Вычисляем градиенты h(x, y) в текущей точке
        grad_x_h, grad_y_h = compute_grad_h(x, y, data, d1, d2)
        # Получаем размеры данных и число кластеров
        m, n = data.shape
        k = y.shape[0]
        # Определяем новые переменные для оптимизации
        new_x = cp.Variable((m, k))
        new_y = cp.Variable((k, n))
        # Целевая функция: g(x, y) - <grad_h, (x, y)>
        objective = cp.Minimize(
            compute_g(new_x, new_y, data, d1, d2) -
            cp.sum(cp.multiply(grad_x_h, new_x)) -
            cp.sum(cp.multiply(grad_y_h, new_y))
        )
        # Ограничения: сумма x по строкам = 1, x >= 0, x <= 1
        constraints = [cp.sum(new_x, axis=1) == 1, new_x >= 0, new_x <= 1]

        if tau is not None:
            for i in range(k):
                constraints.append(cp.sum(new_x[:, i]) >= tau)

        # Формируем задачу оптимизации
        prob = cp.Problem(objective, constraints)
        # Решаем задачу с помощью солвера CLARABEL
        prob.solve(solver='CLARABEL', verbose=False)
        # Получаем новые значения x и y
        new_x_val = new_x.value
        new_y_val = new_y.value
        # Проверяем, решена ли задача и достаточно ли мала разница
        if (new_x_val is None or new_y_val is None or
                (np.linalg.norm(new_x_val - x) < tol and np.linalg.norm(new_y_val - y) < tol)):
            break
        x = new_x_val
        y = new_y_val
    return x, y

# Основной алгоритм Global Search Scheme (GSS)
def local_search_scheme(data, k, d1=0.1, d2=0.55,tau=None, x0=None, y0=None):
    m, n = data.shape
    if x0 is None or y0 is None:
        x = np.random.rand(m, k)
        x /= x.sum(axis=1, keepdims=True)
        y = data[np.random.choice(m, k, replace=False)]
    else:
        x = x0.copy()
        y = y0.copy()
    x, y = dca_local_search(x, y, data, d1, d2, tau)
    return x, y

def tune_d1_d2(data, k, d1_values, d2_values):
    best_score = -1
    best_params = (None, None)

    for d1, d2 in itertools.product(d1_values, d2_values):
        x_opt, y_opt = local_search_scheme(data, k, d1, d2)
        labels = np.argmax(x_opt, axis=1)

        # Silhouette требует хотя бы 2 кластера в данных
        if len(set(labels)) > 1:
            score = silhouette_score(data, labels)
            print(f"d1={d1}, d2={d2}, silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_params = (d1, d2)

    print("\nЛучшие параметры:")
    print(f"d1 = {best_params[0]}, d2 = {best_params[1]}, silhouette = {best_score:.4f}")
    return best_params

# Пример использования с сохранением данных и результатов
if __name__ == "__main__":
    print(cp.installed_solvers())
    np.random.seed(42)

    # 600 точек, 25 кластеров
    data = np.random.rand(600, 2)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_df = pd.DataFrame(data, columns=['x', 'y'])
    data_df.to_csv('data.csv', index=False)
    print("Исходные данные сохранены в 'data.csv'")

    k = 25
    d1 = 0.5
    d2 = 0.5

    start = time.time()
    x_opt, y_opt = local_search_scheme(data, k, d1, d2)
    elapsed = time.time() - start

    # Метки кластеров
    cluster_labels = np.argmax(x_opt, axis=1)

    # Silhouette Score
    if len(set(cluster_labels)) > 1:
        sil_score = silhouette_score(data, cluster_labels)
    else:
        sil_score = float('nan')

    # Сохранение результатов
    result_df = pd.DataFrame(data, columns=['x', 'y'])
    result_df['cluster'] = cluster_labels
    result_df.to_csv('clusters.csv', index=False)

    # Вывод статистики
    print(f"\nВремя работы: {elapsed:.3f} сек.")
    print(f"Индекс силуэта: {sil_score:.4f}")
    print("Центры кластеров:\n", y_opt)

    counts = np.sum(x_opt > 0.5, axis=0)
    for i, cnt in enumerate(counts, start=1):
        print(f"Кластер {i}: {cnt} объектов")