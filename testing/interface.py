import unittest
from unittest.mock import patch, MagicMock
import tkinter as tk
from GUI import App  # Обнови путь при необходимости
import numpy as np

class TestAppGUI(unittest.TestCase):

    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Не отображать окно
        self.app = App(self.root)
        self.app.data_processor.get_selected_data = MagicMock(return_value=np.array([[1, 2], [3, 4]]))
        self.app.data_processor.num_samples = 2
        self.app.feature_vars = [tk.BooleanVar(value=True), tk.BooleanVar(value=False)]
        self.app.data_processor.num_classes = 2  # для автоподбора

    def tearDown(self):
        self.root.destroy()

    @patch('tkinter.filedialog.askopenfilename')
    @patch('GUI.DataProcessor.load_data')
    def test_load_valid_txt_file(self, mock_load_data, mock_open):
        mock_open.return_value = "test_data.txt"
        mock_load_data.return_value = (True, "Файл загружен успешно")
        self.app.load_button.invoke()

        self.assertIn("test_data.txt", self.app.file_label.cget("text"))
        self.assertEqual(self.app.status_label.cget("text"), "Файл загружен успешно")

    @patch('tkinter.filedialog.askopenfilename')
    @patch('GUI.DataProcessor.load_data')
    def test_load_invalid_file_format(self, mock_load_data, mock_open):
        mock_open.return_value = "invalid.jpg"
        mock_load_data.return_value = (False, "Ошибка: неверный формат")
        with patch('tkinter.messagebox.showerror') as mock_error:
            self.app.load_button.invoke()
            mock_error.assert_called_once()
            self.assertIn("Ошибка", mock_error.call_args[0][0])

    def test_run_without_features(self):
        for var in self.app.feature_vars:
            var.set(False)
        with patch('tkinter.messagebox.showerror') as mock_error:
            self.app.run_button.invoke()
            mock_error.assert_called_once_with("Ошибка", "Выберите хотя бы один признак")

    def test_invalid_cluster_input(self):
        self.app.clusters_entry.insert(0, "abc")
        with patch('tkinter.messagebox.showerror') as mock_error:
            self.app.run_button.invoke()
            mock_error.assert_called_once_with("Ошибка", "Количество кластеров должно быть целым числом")

    @patch('GUI.KMeansClustering.fit_predict', return_value=[0, 1])
    @patch('GUI.Visualizer.plot_clusters')
    @patch('GUI.MetricsCalculator.calculate_all', return_value={
        "Инерция": 120.5,
        "Индекс силуэта": 0.75,
        "Индекс Давида-Болдуина": 0.43
    })
    def test_run_with_empty_cluster_entry(self, mock_metrics, mock_plot, mock_fit):
        self.app.clusters_entry.delete(0, tk.END)

        with patch.object(self.app.metrics_text, 'insert') as mock_insert:
            self.app.run_button.invoke()

            mock_fit.assert_called_once()
            mock_metrics.assert_called_once()
            mock_plot.assert_called_once()
            mock_insert.assert_called()  # Проверяем, что метрики были выведены


    def test_clear_reference_table(self):
        self.app.data_processor.reference_labels = [0, 1]
        self.app.ref_table.insert('', 'end', values=[1.0, 2.0, 0])
        self.app.clear_ref_button.config(state=tk.NORMAL)

        with patch('tkinter.messagebox.showinfo') as mock_info:
            self.app.clear_reference_data()
            self.assertEqual(self.app.ref_table.get_children(), ())
            self.assertIsNone(self.app.data_processor.reference_labels)
            mock_info.assert_called_once()

    @patch('tkinter.filedialog.asksaveasfilename')
    def test_save_results(self, mock_save):
        mock_save.return_value = "output.txt"
        self.app.metrics_result = {'Silhouette': 0.75}
        self.app.data = np.array([[1, 2], [3, 4]])
        self.app.cluster_labels = [0, 1]

        with patch('builtins.open', new_callable=unittest.mock.mock_open) as mock_file:
            with patch('tkinter.messagebox.showinfo') as mock_info:
                self.app.export_button.invoke()
                mock_info.assert_called()

    def test_window_close(self):
        with patch.object(self.root, 'destroy') as mock_destroy:
            self.root.destroy()
            mock_destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
