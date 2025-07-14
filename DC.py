import numpy as np
import cvxpy as cp


# Функция для вычисления выпуклой функции g(x, y)
def compute_g(x, y, data, d1, d2):
    """Вычисляет значение функции g(x, y) для задачи MSSC."""
    m, n = data.shape  # m - кол-во точек данных, n - размерность точек
    k = y.shape[0]  # k - количество кластеров
    g_val = 0
    for i in range(k):
        for j in range(m):
            # Вычисляем квадрат евклидова расстояния между центром кластера и точкой
            dist = np.linalg.norm(y[i] - data[j]) ** 2
            # Суммируем слагаемые: d1 * расстояние + d2 * квадрат переменной присваивания
            g_val += d1 * dist + d2 * x[j, i] ** 2
    return g_val


# Функция для вычисления выпуклой функции h(x, y)
def compute_h(x, y, data, d1, d2):
    """Вычисляет значение функции h(x, y) для задачи MSSC."""
    m, n = data.shape  # m - кол-во точек, n - размерность
    k = y.shape[0]  # k - количество кластеров
    h_val = 0
    for i in range(k):
        for j in range(m):
            # Вычисляем квадрат евклидова расстояния
            dist = np.linalg.norm(y[i] - data[j]) ** 2
            # Суммируем слагаемые: d1 * расстояние + d2 * x^2 - x * расстояние
            h_val += d1 * dist + d2 * x[j, i] ** 2 - x[j, i] * dist
    return h_val


# Функция для вычисления градиентов функции h(x, y)
def compute_grad_h(x, y, data, d1, d2):
    """Вычисляет градиенты функции h(x, y) по x и y."""
    m, n = data.shape  # m - кол-во точек, n - размерность
    k = y.shape[0]  # k - количество кластеров
    grad_y = np.zeros_like(y)  # Градиент по y (центры кластеров)
    grad_x = np.zeros_like(x)  # Градиент по x (переменные присваивания)
    for i in range(k):
        for j in range(m):
            dist_vec = y[i] - data[j]  # Вектор разности между центром и точкой
            # Градиент по y: 2 * (d1 - x) * (y - a)
            grad_y[i] += 2 * (d1 - x[j, i]) * dist_vec
            # Градиент по x: 2 * d2 * x - ||y - a||^2
            grad_x[j, i] = 2 * d2 * x[j, i] - np.linalg.norm(dist_vec) ** 2
    return grad_x, grad_y


# Локальный поиск с использованием DC Algorithm (DCA)
def dca_local_search(initial_x, initial_y, data, d1, d2, tol=1e-5, max_iter=100):
    """Локальный поиск с использованием DC Algorithm для нахождения локального минимума."""
    x = initial_x.copy()  # Копия начальных переменных присваивания
    y = initial_y.copy()  # Копия начальных центров кластеров
    for _ in range(max_iter):
        # Вычисляем градиенты h в текущей точке
        grad_x_h, grad_y_h = compute_grad_h(x, y, data, d1, d2)

        # Решаем линеаризованную выпуклую подзадачу с помощью CVXPY
        m, n = data.shape
        k = y.shape[0]
        new_x = cp.Variable((m, k))  # Новые переменные присваивания
        new_y = cp.Variable((k, n))  # Новые центры кластеров
        # Целевая функция: g(x, y) - <grad_h, (x, y)>
        objective = cp.Minimize(
            compute_g(new_x, new_y, data, d1, d2) -
            cp.sum(cp.multiply(grad_x_h, new_x)) -
            cp.sum(cp.multiply(grad_y_h, new_y))
        )
        # Ограничения: каждая точка присваивается ровно одному кластеру
        constraints = [cp.sum(new_x, axis=1) == 1, new_x >= 0, new_x <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        # Обновляем решение
        new_x_val = new_x.value
        new_y_val = new_y.value

        # Проверяем условие остановки (конвергенция)
        if (new_x_val is None or new_y_val is None or
                (np.linalg.norm(new_x_val - x) < tol and np.linalg.norm(new_y_val - y) < tol)):
            break
        x = new_x_val
        y = new_y_val
    return x, y


# Проверка условий глобальной оптимальности
def check_global_optimality(x, y, data, d1, d2):
    """Проверяет условия глобальной оптимальности и возвращает новый начальный пункт, если текущий не оптимален."""
    # Вычисляем текущее значение функции и градиенты
    f_val = compute_g(x, y, data, d1, d2) - compute_h(x, y, data, d1, d2)
    grad_x_h, grad_y_h = compute_grad_h(x, y, data, d1, d2)

    # Решаем линеаризованную задачу для проверки условия (E)
    m, n = data.shape
    k = y.shape[0]
    u_x = cp.Variable((m, k))  # Переменные u_x для проверки
    u_y = cp.Variable((k, n))  # Переменные u_y для проверки
    objective = cp.Minimize(compute_g(u_x, u_y, data, d1, d2))
    # Ограничения, включая нарушение условия (E)
    constraints = [
        cp.sum(u_x, axis=1) == 1,
        u_x >= 0,
        u_x <= 1,
        compute_g(u_x, u_y, data, d1, d2) -
        (cp.sum(cp.multiply(grad_x_h, u_x - x)) + cp.sum(cp.multiply(grad_y_h, u_y - y))) <
        compute_h(x, y, data, d1, d2)
    ]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if prob.status == cp.OPTIMAL:
        # Условие (E) нарушено, возвращаем новый начальный пункт
        return False, (u_x.value, u_y.value)
    # Условие (E) выполнено, текущая точка может быть глобальным минимумом
    return True, None


# Основной алгоритм Global Search Scheme (GSS)
def global_search_scheme(data, k, d1=1.0, d2=1.0, max_iter=10):
    """Реализует Global Search Scheme для задачи MSSC."""
    m, n = data.shape  # m - кол-во точек, n - размерность

    # Инициализация переменных присваивания и центров кластеров
    x = np.random.rand(m, k)  # Случайные значения для x
    x /= x.sum(axis=1, keepdims=True)  # Нормализация, чтобы сумма по строкам равнялась 1
    y = data[np.random.choice(m, k, replace=False)]  # Случайные начальные центры

    for _ in range(max_iter):
        # Шаг 1: Локальный поиск с помощью DCA
        x, y = dca_local_search(x, y, data, d1, d2)

        # Шаг 2: Проверка глобальной оптимальности и выход из локального минимума
        is_global, new_start = check_global_optimality(x, y, data, d1, d2)
        if is_global:
            break
        if new_start is not None:
            x, y = new_start  # Обновляем начальную точку для следующей итерации

    return x, y


# Пример использования
if __name__ == "__main__":
    # Генерируем синтетические данные
    np.random.seed(42)
    data = np.random.rand(100, 2)  # 100 точек в 2D-пространстве
    k = 3  # Количество кластеров
    d1, d2 = 1.0, 1.0  # Константы для DC-разложения

    # Запускаем GSS
    x_opt, y_opt = global_search_scheme(data, k, d1, d2)
    print("Центры кластеров:\n", y_opt)
    print("Присваивания (первые 5 строк):\n", x_opt[:5])