import numpy as np

class DataProcessor:
    def __init__(self):
        self.data = None
        self.num_features = 0
        self.num_samples = 0
        self.num_classes = None
        self.reference_labels = None

    def clear_reference_data(self):
        """Очищает эталонные метки"""
        self.reference_labels = None
        self.num_classes = None

    def load_data(self, filepath):
        try:
            with open(filepath, 'r') as file:
                first_line = file.readline().strip()
                parts = first_line.split()

                if len(parts) not in [2, 3]:
                    return False, "Первая строка должна содержать 2 или 3 числа"

                self.num_features = int(parts[0])
                self.num_samples = int(parts[1])
                self.num_classes = int(parts[2]) if len(parts) == 3 else None

                data = []
                labels = []
                has_labels = self.num_classes is not None

                for line in file:
                    row = list(map(float, line.strip().split()))
                    if has_labels:
                        if len(row) != self.num_features + 1:
                            return False, f"Ожидается {self.num_features + 1} значений (признаки + метка), получено {len(row)}"
                        data.append(row[:-1])
                        labels.append(int(row[-1]))
                    else:
                        if len(row) != self.num_features:
                            return False, f"Ожидается {self.num_features} признаков, получено {len(row)}"
                        data.append(row)

                if len(data) != self.num_samples:
                    return False, f"Ожидается {self.num_samples} строк, получено {len(data)}"

                self.data = np.array(data)
                if has_labels:
                    self.reference_labels = np.array(labels)
                    return True, f"Данные успешно загружены (с метками классов, {self.num_classes} классов)"
                return True, "Данные успешно загружены (без меток классов)"

        except Exception as e:
            return False, f"Ошибка при загрузке файла: {str(e)}"

    def load_reference_data(self, filepath):
        try:
            with open(filepath, 'r') as file:
                labels = []
                for line in file:
                    labels.append(int(line.strip()))

                if len(labels) != self.num_samples:
                    return False, f"Количество эталонных меток ({len(labels)}) не совпадает с количеством образцов ({self.num_samples})"

                self.reference_labels = np.array(labels)
                return True, "Эталонные данные успешно загружены"

        except Exception as e:
            return False, f"Ошибка при загрузке эталонных данных: {str(e)}"

    def get_selected_data(self, feature_indices):
        return self.data[:, feature_indices]

