import numpy as np


# --- GSA ---
# x_gsa, y_gsa = global_search_scheme_simple(
#     data, k, d1=1.0, d2=1.0, max_iter=8, restarts_perturb=20
# )
# labels_gsa = np.argmin(np.sum((data[:, None, :] - y_gsa[None, :, :]) ** 2, axis=2), axis=1)
# obj_gsa = objective_value(data, y_gsa)
# p_gsa, r_gsa, f_gsa = pairwise_metrics(true_labels, labels_gsa)

def objective_value(data, centers):
    dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2) ** 2
    return np.sum(np.min(dists, axis=1))

def project_simplex(v):
    """Проекция вектора v на симплекс {x>=0, sum=1} (алгоритм Duchi et al.)"""
    n = v.shape[0]
    if np.all(v == v[0]):
        return np.ones_like(v) / n
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

# -----------------------
# Градиенты h (numpy-версия)
# -----------------------
def compute_grad_h_numpy(x, y, data, d1, d2):
    # x: (m,k), y: (k,n), data: (m,n)
    m, n = data.shape
    k = y.shape[0]
    dist_vec = y[:, None, :] - data[None, :, :]  # (k, m, n)
    coeff = 2 * (d1 - x.T)  # (k,m)
    grad_y = np.sum(coeff[:, :, None] * dist_vec, axis=1)  # (k,n)
    dists_sq = np.sum(dist_vec ** 2, axis=2)  # (k,m)
    grad_x = 2 * d2 * x - dists_sq.T  # (m,k)
    return grad_x, grad_y

# -----------------------
# Упрощённый DCA local search
# -----------------------
def dca_local_search_simplified(initial_x, initial_y, data, d1, d2, tol=1e-5, max_iter=200):
    x = initial_x.copy()
    y = initial_y.copy()
    m, n = data.shape
    k = y.shape[0]
    sum_a = np.sum(data, axis=0)  # (n,)
    for it in range(max_iter):
        grad_x_h, grad_y_h = compute_grad_h_numpy(x, y, data, d1, d2)
        # x update: x = grad_x_h / (2*d2) projected on simplex for each point
        x_new = np.zeros_like(x)
        for j in range(m):
            v = grad_x_h[j, :] / (2.0 * d2)
            x_new[j, :] = project_simplex(v)
        # y update: y_i = (sum_j a_j + grad_y_h_i/(2*d1)) / m
        y_new = np.zeros_like(y)
        for i in range(k):
            y_new[i, :] = (sum_a + grad_y_h[i, :] / (2.0 * d1)) / m
        if np.linalg.norm(x_new - x) < tol and np.linalg.norm(y_new - y) < tol:
            x, y = x_new, y_new
            break
        x, y = x_new, y_new
    return x, y

# -----------------------
# Простая версия Global Search Scheme (GSA-like)
# -----------------------
def global_search_scheme_simple(data, k, d1=1.0, d2=1.0, max_iter=6, restarts_perturb=10):
    m, n = data.shape
    x = np.random.rand(m, k)
    x = x / x.sum(axis=1, keepdims=True)
    idx = np.random.choice(m, k, replace=False)
    y = data[idx].copy()
    best_obj = np.inf
    best_xy = (x, y)
    for iteration in range(max_iter):
        x, y = dca_local_search_simplified(x, y, data, d1, d2)
        obj = objective_value(data, y)
        if obj < best_obj:
            best_obj = obj
            best_xy = (x.copy(), y.copy())
        improved = False
        sigma = np.std(data, axis=0).mean()
        for r in range(restarts_perturb):
            y_try = best_xy[1] + np.random.randn(*best_xy[1].shape) * (0.5 * sigma)
            x0 = np.random.rand(m, k); x0 = x0 / x0.sum(axis=1, keepdims=True)
            x_try, y_try = dca_local_search_simplified(x0, y_try, data, d1, d2)
            obj_try = objective_value(data, y_try)
            if obj_try < best_obj - 1e-8:
                best_obj = obj_try
                best_xy = (x_try.copy(), y_try.copy())
                x, y = x_try, y_try
                improved = True
                break
        if not improved:
            break
    return best_xy[0], best_xy[1]