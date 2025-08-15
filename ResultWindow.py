import numpy as np
from PySide6 import QtWidgets
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QPushButton, QMessageBox, QFileDialog, QDialog, QTextEdit, QWidgetAction, QLabel
)
from PySide6.QtCore import Qt

class ResultsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result_counter = 0
        self.setWindowTitle("Результаты")
        self.resize(900, 800)

        central = QWidget()
        self._create_toolbar()
        self.setCentralWidget(central)

        # Сначала создаём layout
        self.main_layout = QVBoxLayout(central)

        # Прокручиваемая область
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.main_layout.addWidget(scroll)

        # Контейнер внутри scroll_area
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_layout.setSpacing(5)
        scroll.setWidget(self.scroll_content)

    def _create_toolbar(self):
        self.toolbar = self.addToolBar("Help")
        self.toolbar.setFloatable(False)
        self.toolbar.setMovable(False)
        self.toolbar.setVisible(True)

        # Создаем заголовок
        title_label = QLabel("Результат кластеризации методом k-средних / DC (LSA)")
        font = title_label.font()
        font.setPointSize(12)
        font.setBold(True)
        title_label.setFont(font)

        # Добавляем заголовок в тулбар через QWidgetAction
        title_action = QWidgetAction(self)
        title_action.setDefaultWidget(title_label)
        self.toolbar.addAction(title_action)

        # Добавляем spacer, чтобы кнопка вопроса была справа
        spacer = QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # Кнопка помощи
        self.help_action = QAction("?", self)
        self.help_action.triggered.connect(self.show_help)
        self.toolbar.addAction(self.help_action)

        # Стиль для кнопки (поскольку QAction отображается как кнопка в тулбаре)
        self.toolbar.setStyleSheet("""
                    QToolBar {
                        background: lightblue;  /* Фон тулбара для видимости */
                    }
                    QToolButton {  /* Стиль для кнопки от QAction */
                        background-color: white;
                        font-family: "Arial";   /* укажите ваш шрифт */
                        font-size: 16pt;        /* размер шрифта */
                        padding: 5px;
                        border: 1px solid gray;
                    }
                    QToolButton:hover {
                        background-color: #90EE90;
                    }
                """)

    def show_help(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Руководство пользователя")
        dialog.setModal(True)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        try:
            with open("result_guide.txt", 'r', encoding='utf-8') as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = "Файл руководства 'result_guide.txt' не найден."
        except Exception as e:
            help_text = f"Ошибка при загрузке руководства: {str(e)}"

        text_edit.setText(help_text)
        layout.addWidget(text_edit)

        dialog.resize(500, 400)
        dialog.exec_()

    def add_result(self, plot_widget: QWidget, metrics: dict, title: str, data: np.ndarray = None,
                   cluster_labels: np.ndarray = None, fixed_height=400):
        """
        Добавляет блок с графиком и таблицей метрик.
        Сохраняет data и cluster_labels для возможности экспорта.
        """
        self.result_counter += 1
        numbered_title = f"{self.result_counter}. {title}"

        group = QGroupBox(numbered_title)
        group.setMinimumHeight(fixed_height)
        group.setMaximumHeight(fixed_height)

        layout = QHBoxLayout(group)
        layout.addWidget(plot_widget, 2)

        # Таблица метрик
        table = QTableWidget()
        table.setColumnCount(2)
        table.setRowCount(len(metrics))
        table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        table.verticalHeader().setVisible(False)
        for i, (k, v) in enumerate(metrics.items()):
            table.setItem(i, 0, QTableWidgetItem(str(k)))
            table.setItem(i, 1, QTableWidgetItem(f"{v:.6f}" if isinstance(v, (int, float)) else str(v)))
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        # Правый контейнер: таблица + кнопка
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.addStretch(1)
        right_layout.addWidget(table, alignment=Qt.AlignCenter)

        # Кнопка экспорта для этого блока
        if data is not None and cluster_labels is not None:
            export_btn = QPushButton(f"Скачать кластеры ({self.result_counter})")

            def export_block_clusters():
                # Составляем имя файла по шаблону
                num_samples = data.shape[0]
                num_clusters = len(np.unique(cluster_labels))
                algo_name = title.replace(" ", "_")  # используем title блока как метод
                default_fname = f"{algo_name}_{num_clusters}_{num_samples}.txt"

                fname, _ = QFileDialog.getSaveFileName(
                    self, "Сохранить кластеры", default_fname, "Текстовый файл (*.txt)"
                )
                if not fname:
                    return
                try:
                    num_features = data.shape[1]

                    with open(fname, 'w', encoding='utf-8') as f:
                        # Первая строка: признаки, объекты, кластеры
                        f.write(f"{num_features} {num_samples} {num_clusters}\n")
                        # Основные данные: признаки + номер кластера (целый)
                        for row, cluster in zip(data, cluster_labels):
                            row_str = " ".join(f"{v:.6f}" for v in row)
                            f.write(f"{row_str} {int(cluster)}\n")

                    QMessageBox.information(self, "Экспорт", f"Файл сохранён: {fname}")
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")

            export_btn.clicked.connect(export_block_clusters)
            right_layout.addWidget(export_btn, alignment=Qt.AlignCenter)

        right_layout.addStretch(1)
        layout.addWidget(right_container, 1)

        self.scroll_layout.addWidget(group)

    def add_export_clusters_button(self):
        # Кнопка для сохранения кластеров
        self.export_btn = QPushButton("Скачать кластеры")
        self.export_btn.clicked.connect(self.export_clusters)
        self.layout().addWidget(self.export_btn)

    def export_clusters(self):
        if not hasattr(self, 'cluster_labels') or not hasattr(self, 'data'):
            QMessageBox.warning(self, "Экспорт", "Нет данных для экспорта")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Сохранить кластеры", "", "Текстовый файл (*.txt)"
        )
        if not fname:
            return

        try:
            num_features = self.data.shape[1]
            num_samples = self.data.shape[0]
            num_clusters = len(np.unique(self.cluster_labels))

            # Первая строка с мета-информацией
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(f"{num_features} {num_samples} {num_clusters}\n")
                # Основные данные: признаки + номер кластера (целый)
                for row, cluster in zip(self.data, self.cluster_labels):
                    row_str = " ".join(f"{v:.6f}" for v in row)
                    f.write(f"{row_str} {int(cluster)}\n")

            QMessageBox.information(self, "Экспорт", f"Файл сохранён: {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {e}")