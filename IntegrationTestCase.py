import unittest
import tkinter as tk
from unittest.mock import patch, MagicMock
from GUI import App  # путь к файлу, содержащему ваш GUI-класс
import numpy as np

class TestAppIntegration(unittest.TestCase):
    def setUp(self):
        print("\n[Подготовка] Инициализация приложения")
        self.root = tk.Tk()
        self.root.withdraw()  # скрываем окно во время теста
        self.app = App(self.root)

    def tearDown(self):
        print("[Завершение]")
        # Если у тебя есть объект с подключением к БД, закрывай его
        if hasattr(self.app, 'db_connection'):
            self.app.db_connection.close()
        self.root.destroy()


    @patch("tkinter.filedialog.askopenfilename")
    def test_load_file_success_and_run_clustering(self, mock_file_dialog):
        print("[Тест] Загрузка файла и запуск кластеризации")
        mock_file_dialog.return_value = "dummy_file.txt"

        dummy_data = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]])
        dummy_labels = [0, 1, 1]

        self.app.data_processor.load_data = MagicMock(return_value=(True, "Данные успешно загружены"))
        self.app.data_processor.get_selected_data = MagicMock(return_value=dummy_data)
        self.app.data_processor.data = dummy_data
        self.app.data_processor.num_features = 2
        self.app.data_processor.num_samples = 3

        self.app.load_file()
        self.app.root.update()

        print(f"[Проверка] Создано чекбоксов: {len(self.app.feature_vars)}")
        self.assertEqual(len(self.app.feature_vars), 2)
        for var in self.app.feature_vars:
            self.assertTrue(var.get())

        self.app.clustering.fit_predict = MagicMock(return_value=dummy_labels)
        self.app.visualizer.plot_clusters = MagicMock()
        self.app.visualizer.plot_reference_clusters = MagicMock()
        self.app.metrics_calculator.calculate_all = MagicMock(return_value={"Silhouette": 0.75})

        self.app.run_clustering()
        self.app.root.update()

        metrics_output = self.app.metrics_text.get(1.0, tk.END)
        print(f"[Результат] Метрики:\n{metrics_output.strip()}")
        self.assertIn("Silhouette", metrics_output)
        self.assertIn("0.750000", metrics_output)

    @patch("tkinter.messagebox.showerror")
    def test_run_clustering_with_no_selected_features(self, mock_showerror):
        print("[Тест] Запуск кластеризации без выбранных признаков")
        self.app.feature_vars = [tk.BooleanVar(value=False) for _ in range(2)]

        self.app.run_clustering()
        self.app.root.update()

        print("[Проверка] Ожидание вызова ошибки при отсутствии признаков")
        mock_showerror.assert_called_once_with("Ошибка", "Выберите хотя бы один признак")

    @patch("tkinter.filedialog.askopenfilename")
    @patch("tkinter.messagebox.showinfo")
    def test_load_reference_success(self, mock_info, mock_dialog):
        print("[Тест] Загрузка эталонных меток")
        mock_dialog.return_value = "reference.txt"
        ref_content = "2 3 2\n1.0 2.0 0\n1.1 2.1 1\n0.9 1.9 1\n"
        with patch("builtins.open", new_callable=unittest.mock.mock_open, read_data=ref_content):
            self.app.data_processor.num_samples = 3
            self.app.load_reference()

        self.app.root.update()
        print(f"[Проверка] Загружено эталонных меток: {len(self.app.data_processor.reference_labels)}")
        self.assertIsNotNone(self.app.data_processor.reference_labels)
        self.assertEqual(len(self.app.data_processor.reference_labels), 3)
        mock_info.assert_called()

    @patch("tkinter.filedialog.askopenfilename")
    @patch("tkinter.messagebox.showerror")
    def test_load_file_with_invalid_format(self, mock_showerror, mock_file_dialog):
        print("[Тест] Загрузка файла с неверным форматом")

        mock_file_dialog.return_value = "bad_file.txt"
        self.app.data_processor.load_data = MagicMock(return_value=(False, "Ошибка формата файла"))
        print("[Проверка] Статус: Ошибка формата файла")
        self.app.load_file()
        self.app.root.update()

        mock_showerror.assert_called_once_with("Ошибка", "Ошибка формата файла")

    @patch("tkinter.filedialog.askopenfilename")
    @patch("tkinter.messagebox.showerror")
    def test_load_file_missing_first_line(self, mock_showerror, mock_file_dialog):
        print("[Тест] Загрузка файла без первой строки")

        mock_file_dialog.return_value = "bad_file.txt"

        # Заглушаем load_data чтобы вернуть ошибку
        self.app.data_processor.load_data = MagicMock(return_value=(False, "Первая строка отсутствует или некорректна"))

        self.app.load_file()
        self.app.root.update()

        # Проверяем, что статус в статусном лейбле содержит ошибку
        status_text = self.app.status_label.cget("text")
        print(f"[Проверка] Статус: {status_text}")
        self.assertEqual(status_text, "Первая строка отсутствует или некорректна")


if __name__ == "__main__":
    unittest.main()
