import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QSize
from DataManager import DataProcessor
from ClusterEngine import KMeansClustering
from Visualization import Visualizer
from Metrics import MetricsCalculator
from DBHandler import Database

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ кластеризации данных")
        self.setGeometry(100, 100, 700, 750)

        # Инициализация компонентов
        self.data_processor = DataProcessor()
        self.clustering = KMeansClustering()
        self.visualizer = Visualizer()
        self.metrics_calculator = MetricsCalculator()
        self.database = Database()
        self.feature_vars = []  # Для хранения состояний чекбоксов

        # Главный виджет и компоновка
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Заголовок
        self.header = QLabel("Анализ кластеризации методом k-средних", self)
        self.header.setStyleSheet("font: bold 11pt Helvetica; padding: 5px;")
        self.layout.addWidget(self.header)

        # Секция загрузки данных
        self.load_frame = QWidget()
        load_layout = QHBoxLayout(self.load_frame)
        self.load_button = QPushButton("Выбрать файл", self)
        self.load_button.clicked.connect(self.load_file)
        self.file_label = QLabel("Файл не выбран", self)
        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: green;")
        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.file_label)
        load_layout.addWidget(self.status_label)
        self.layout.addWidget(self.load_frame)

        # Секция выбора признаков с прокруткой
        self.features_scroll = QScrollArea()
        self.features_widget = QWidget()
        self.features_layout = QVBoxLayout(self.features_widget)
        self.features_scroll.setWidget(self.features_widget)
        self.features_scroll.setWidgetResizable(True)
        self.features_scroll.setFixedHeight(50)
        self.layout.addWidget(self.features_scroll)

        # Параметры кластеризации
        self.params_frame = QWidget()
        params_layout = QHBoxLayout(self.params_frame)
        self.clusters_label = QLabel("Количество кластеров (оставьте пустым для автоматического выбора):", self)
        self.clusters_entry = QLineEdit(self)
        self.clusters_entry.setFixedWidth(100)
        params_layout.addWidget(self.clusters_label)
        params_layout.addWidget(self.clusters_entry)
        self.layout.addWidget(self.params_frame)

        # Таблицы данных
        self.tables_frame = QWidget()
        tables_layout = QHBoxLayout(self.tables_frame)
        self.data_table_frame = QWidget()
        data_table_layout = QVBoxLayout(self.data_table_frame)
        self.data_table = QTableWidget(6, 0)  # Фиксированная высота 6 строк
        data_table_layout.addWidget(self.data_table)
        tables_layout.addWidget(self.data_table_frame)

        self.ref_table_frame = QWidget()
        ref_table_layout = QVBoxLayout(self.ref_table_frame)
        self.ref_table = QTableWidget(6, 0)  # Фиксированная высота 6 строк
        ref_table_layout.addWidget(self.ref_table)
        tables_layout.addWidget(self.ref_table_frame)
        self.layout.addWidget(self.tables_frame)

        # Метрики
        self.metrics_frame = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_frame)
        self.metrics_text = QTextEdit(self)
        self.metrics_text.setFixedHeight(150)
        self.metrics_text.setFontFamily("Consolas")
        self.metrics_text.setFontPointSize(9)
        metrics_layout.addWidget(self.metrics_text)
        self.layout.addWidget(self.metrics_frame)

        # Панель действий
        self.actions_frame = QWidget()
        actions_layout = QHBoxLayout(self.actions_frame)
        self.ref_button = QPushButton("Загрузить эталонные данные", self)
        self.ref_button.clicked.connect(self.load_reference)
        self.clear_ref_button = QPushButton("Очистить таблицу с эталонными данными", self)
        self.clear_ref_button.clicked.connect(self.clear_reference_data)
        self.clear_ref_button.setEnabled(False)
        self.export_button = QPushButton("Сохранить результаты", self)
        self.export_button.clicked.connect(self.export_results)
        actions_layout.addWidget(self.ref_button)
        actions_layout.addWidget(self.clear_ref_button)
        actions_layout.addWidget(self.export_button)
        self.layout.addWidget(self.actions_frame)

        # Кнопка запуска
        self.run_button = QPushButton("Визуализировать кластеры", self)
        self.run_button.clicked.connect(self.run_clustering)
        self.layout.addWidget(self.run_button)

        # Установка стилей
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Helvetica;
                font-size: 9pt;
            }
            QPushButton {
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #4a7abc;
                color: black;
            }
            QLabel[error="true"] {
                color: red;
            }
        """)

    def load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл данных",
            "", "Текстовые файлы (*.txt);;CSV файлы (*.csv);;Все файлы (*.*)"
        )
        if filepath:
            self.file_label.setText(filepath.split('/')[-1])
            success, message = self.data_processor.load_data(filepath)

            if success:
                self.status_label.setText(message)
                self.update_features_checkboxes()
                self.update_data_table()
            else:
                self.status_label.setText(message)
                self.status_label.setProperty("error", True)
                self.status_label.style().unpolish(self.status_label)
                self.status_label.style().polish(self.status_label)
                QMessageBox.critical(self, "Ошибка", "Формат файла должен быть txt")

    def update_features_checkboxes(self):
        # Очищаем предыдущие чекбоксы
        for i in reversed(range(self.features_layout.count())):
            self.features_layout.itemAt(i).widget().setParent(None)
        self.feature_vars.clear()

        # Создаем чекбоксы для каждого признака
        for i in range(self.data_processor.num_features):
            checkbox = QPushButton(f"Признак {i + 1}", self)
            checkbox.setCheckable(True)
            checkbox.setChecked(True)
            checkbox.clicked.connect(lambda checked, idx=i: self.toggle_feature(idx, checked))
            self.feature_vars.append(checkbox)
            self.features_layout.addWidget(checkbox)

    def toggle_feature(self, idx, checked):
        self.feature_vars[idx].setChecked(checked)

    def run_clustering(self):
        selected_indices = [i for i, cb in enumerate(self.feature_vars) if cb.isChecked()]

        if not selected_indices:
            QMessageBox.critical(self, "Ошибка", "Выберите хотя бы один признак")
            return

        n_clusters = self.clusters_entry.text()
        try:
            n_clusters = int(n_clusters) if n_clusters else None
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Количество кластеров должно быть целым числом")
            return

        data = self.data_processor.get_selected_data(selected_indices)

        try:
            if hasattr(self.data_processor, 'reference_labels') and self.data_processor.reference_labels is not None:
                self.visualizer.plot_reference_clusters(data, self.data_processor.reference_labels)

            num_classes = getattr(self.data_processor, 'num_classes', None)
            labels = self.clustering.fit_predict(data, n_clusters, num_classes)
            self.visualizer.plot_clusters(data, labels)

            metrics = self.metrics_calculator.calculate_all(data, labels)

            if hasattr(self.data_processor, 'reference_labels') and self.data_processor.reference_labels is not None:
                ref_metrics = self.metrics_calculator.calculate_comparative(
                    self.data_processor.reference_labels, labels
                )
                metrics.update(ref_metrics)

            self.show_metrics(metrics)
            self.cluster_labels = labels
            self.data = data
            self.metrics_result = metrics

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при кластеризации: {str(e)}")

    def show_metrics(self, metrics):
        self.metrics_text.clear()
        self.metrics_text.append("# Результаты кластеризации\n\n")

        max_name_length = max(len(name) for name in metrics.keys())
        for name, value in metrics.items():
            self.metrics_text.append(f"| {name.ljust(max_name_length)} | {value:.6f} |\n")

        self.metrics_text.append("|" + "-" * (max_name_length + 2) + "|" + "-" * 12 + "|\n")

    def load_reference(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с эталонными данными",
            "", "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    first_line = file.readline().strip().split()
                    if len(first_line) != 3:
                        raise ValueError("Файл эталонных данных должен содержать 3 числа в первой строке")

                    num_features, num_samples, num_classes = map(int, first_line)

                    if num_samples != self.data_processor.num_samples:
                        raise ValueError(f"Количество образцов в эталонных данных ({num_samples}) не совпадает с загруженными данными ({self.data_processor.num_samples})")

                    data, labels = [], []
                    for line in file:
                        row = list(map(float, line.strip().split()))
                        if len(row) != num_features + 1:
                            raise ValueError(f"Ожидается {num_features + 1} значений (признаки + метка), получено {len(row)}")
                        data.append(row[:-1])
                        labels.append(int(row[-1]))

                    self.data_processor.reference_data = np.array(data)
                    self.data_processor.reference_labels = np.array(labels)
                    self.data_processor.num_classes = num_classes

                    self.update_ref_table()
                    self.clear_ref_button.setEnabled(True)
                    QMessageBox.information(self, "Успех", f"Эталонные данные успешно загружены ({num_classes} классов)")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить эталонные данные: {str(e)}")

    def save_results(self):
        try:
            QMessageBox.information(self, "Успех", "Результаты сохранены в базу данных")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")

    def clear_reference_data(self):
        self.data_processor.reference_labels = None
        self.clear_ref_button.setEnabled(False)
        self.ref_table.clearContents()
        QMessageBox.information(self, "Успех", "Эталонные данные удалены")

    def update_data_table(self):
        try:
            if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
                return
            if len(self.data_processor.data) == 0:
                return

            num_features = getattr(self.data_processor, 'num_features',
                                   len(self.data_processor.data[0]) if len(self.data_processor.data) > 0 else 0)
            self.data_table.setColumnCount(num_features)
            self.data_table.setHorizontalHeaderLabels([f"Признак {i + 1}" for i in range(num_features)])

            self.data_table.setRowCount(min(6, len(self.data_processor.data)))  # Ограничение до 6 строк
            for row_idx, row in enumerate(self.data_processor.data[:6]):
                for col_idx, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.data_table.setItem(row_idx, col_idx, item)

        except Exception as e:
            print(f"Ошибка при обновлении таблицы данных: {str(e)}")

    def update_ref_table(self):
        if (not hasattr(self.data_processor, 'reference_data') or self.data_processor.reference_data is None or
                not hasattr(self.data_processor, 'reference_labels') or self.data_processor.reference_labels is None):
            return

        num_features = self.data_processor.reference_data.shape[1]
        self.ref_table.setColumnCount(num_features + 1)
        self.ref_table.setHorizontalHeaderLabels([f"Признак {i + 1}" for i in range(num_features)] + ["Метка"])

        self.ref_table.setRowCount(min(6, len(self.data_processor.reference_labels)))  # Ограничение до 6 строк
        for row_idx in range(min(6, len(self.data_processor.reference_labels))):
            for col_idx, value in enumerate(self.data_processor.reference_data[row_idx]):
                item = QTableWidgetItem(str(value))
                self.ref_table.setItem(row_idx, col_idx, item)
            item = QTableWidgetItem(str(self.data_processor.reference_labels[row_idx]))
            self.ref_table.setItem(row_idx, num_features, item)

    def export_results(self):
        try:
            QMessageBox.information(self, "Успех", "Результаты сохранены в базу данных")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "", "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write("=== Метрики кластеризации ===\n")
                    if hasattr(self, 'metrics_result'):
                        for name, value in self.metrics_result.items():
                            f.write(f"{name}: {value}\n")
                    else:
                        f.write("Метрики отсутствуют.\n")

                    f.write("\n=== Объекты и метки кластеров ===\n")
                    if hasattr(self, 'data') and hasattr(self, 'cluster_labels'):
                        import pandas as pd
                        df = pd.DataFrame(self.data)
                        df['cluster'] = self.cluster_labels
                        df.to_csv(f, sep='\t', index=False, header=False)
                    else:
                        f.write("Нет доступных данных или меток для экспорта.\n")

                QMessageBox.information(self, "Успех", "Результаты успешно экспортированы")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())