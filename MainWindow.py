import sys
import threading
import numpy as np
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QSpinBox, QGroupBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QCheckBox, QTextEdit, QMessageBox, QDialog,
    QProgressBar, QWidgetAction
)

# === Ваши модули (оставлены как есть) ===
from DataManager import DataProcessor
from ClusterEngine import KMeansClustering, DCClustering
from ResultWindow import ResultsWindow
from Visualization import Visualizer
from Metrics import MetricsCalculator
from DBHandler import Database

# ---------------------------
# Вспомогательный поток для кластеризации
# ---------------------------
class ClusteringWorker(QtCore.QThread):
    finished_ok = QtCore.Signal(np.ndarray, dict)  # labels, metrics
    failed = QtCore.Signal(str)

    def __init__(self, data, method_name, n_clusters, num_classes, metrics_calculator, parent=None):
        super().__init__(parent)
        self.data = data
        self.method_name = method_name
        self.n_clusters = n_clusters
        self.num_classes = num_classes
        self.metrics_calculator = metrics_calculator

    def run(self):
        try:
            # Выбор алгоритма
            if self.method_name == "KMeans":
                clustering = KMeansClustering()
            elif self.method_name == "DC-алгоритм (LSA)":
                clustering = DCClustering()
            else:
                raise ValueError("Неизвестный метод кластеризации")

            # Запуск кластеризации
            labels = clustering.fit_predict(self.data, self.n_clusters, self.num_classes)

            # Подсчёт метрик
            metrics = self.metrics_calculator.calculate_all(self.data, labels)
            self.finished_ok.emit(labels, metrics)
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------
# Модальное окно загрузки/ожидания
# ---------------------------
class LoadingDialog(QDialog):
    def __init__(self, parent=None, text="Идёт кластеризация..."):
        super().__init__(parent)
        self.setWindowTitle("Подождите…")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(280, 120)

        vbox = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        bar = QProgressBar()
        bar.setRange(0, 0)  # бесконечный индикатор

        vbox.addStretch(1)
        vbox.addWidget(label)
        vbox.addWidget(bar)
        vbox.addStretch(1)

        # Центрирование относительно родителя
        self._center_on_parent()

    def _center_on_parent(self):
        if self.parent() is None:
            return
        self.setGeometry(
            self.parent().geometry().center().x() - self.width() // 2,
            self.parent().geometry().center().y() - self.height() // 2,
            self.width(),
            self.height(),
        )


# ---------------------------
# Основное окно приложения на PySide6
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ кластеризации данных (PySide6)")
        self.resize(820, 700)
        self.results_window = ResultsWindow()

        # Центрирование главного окна
        self._center_window()

        # Увеличенный шрифт по умолчанию
        font = QtGui.QFont()
        font.setPointSize(11)  # общий размер шрифта
        self.setFont(font)

        # Логика/сервисы
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.metrics_calculator = MetricsCalculator()
        self.database = Database()

        self.cluster_labels = None
        self.data = None
        self.metrics_result = {}
        self.feature_checks = []

        # UI
        self._build_ui()
        self._create_toolbar()

    def _center_window(self):
        screen = QApplication.primaryScreen().availableGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def _create_toolbar(self):
        self.toolbar = self.addToolBar("Help")
        self.toolbar.setFloatable(False)
        self.toolbar.setMovable(False)
        self.toolbar.setVisible(True)

        # Создаем заголовок
        title_label = QLabel("Анализ кластеризации методом k-средних / DC (LSA)")
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
            with open("user_guide.txt", 'r', encoding='utf-8') as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = "Файл руководства 'user_guide.txt' не найден."
        except Exception as e:
            help_text = f"Ошибка при загрузке руководства: {str(e)}"

        text_edit.setText(help_text)
        layout.addWidget(text_edit)

        dialog.resize(500, 400)
        dialog.exec_()
    # ---------- UI ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        for btn in self.findChildren(QPushButton):
            btn.setStyleSheet("background-color: lightblue; font-size: 12pt;")

        # Прокручиваемая область
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        v = QVBoxLayout(content)

        # Загрузка данных
        self.grp_load = QGroupBox("Загрузка данных")
        gl = QGridLayout(self.grp_load)
        self.btn_load = QPushButton("Выбрать файл…")
        self.btn_load.clicked.connect(self.on_load_clicked)
        self.lbl_filename = QLabel("Файл не выбран")
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: green;")
        gl.addWidget(self.btn_load, 0, 0)
        gl.addWidget(self.lbl_filename, 0, 1)
        gl.addWidget(self.lbl_status, 1, 0, 1, 2)
        v.addWidget(self.grp_load)

        # Таблицы (данные и эталон)
        tables_box = QHBoxLayout()
        v.addLayout(tables_box)

        self.grp_data = QGroupBox("Загруженные данные")
        self.tbl_data = QTableWidget()
        self.tbl_data.setMinimumHeight(180)
        dl = QVBoxLayout(self.grp_data)
        dl.addWidget(self.tbl_data)
        tables_box.addWidget(self.grp_data, 1)

        self.grp_ref = QGroupBox("Эталонные данные")
        self.tbl_ref = QTableWidget()
        self.tbl_ref.setMinimumHeight(180)
        rl = QVBoxLayout(self.grp_ref)
        rl.addWidget(self.tbl_ref)
        tables_box.addWidget(self.grp_ref, 1)

        # Основной горизонтальный layout для трех блоков
        h = QHBoxLayout()

        # ----------------- Выбор признаков -----------------
        self.grp_features = QGroupBox("Выбор признаков")
        fv = QVBoxLayout(self.grp_features)
        self.scroll_feat = QScrollArea()
        self.scroll_feat.setWidgetResizable(True)
        self.feat_container = QWidget()
        self.feat_layout = QVBoxLayout(self.feat_container)
        self.feat_layout.addStretch(1)
        self.scroll_feat.setWidget(self.feat_container)
        fv.addWidget(self.scroll_feat)

        h.addWidget(self.grp_features)  # добавляем в горизонтальный layout

        # ----------------- Метод кластеризации -----------------
        self.grp_method = QGroupBox("Метод кластеризации")
        ml = QGridLayout(self.grp_method)
        ml.addWidget(QLabel("Метод:"), 0, 0)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["KMeans", "DC-алгоритм (LSA)"])
        ml.addWidget(self.cmb_method, 0, 1)

        h.addWidget(self.grp_method)  # добавляем в горизонтальный layout

        # ----------------- Количество кластеров -----------------
        self.grp_params = QGroupBox("Параметры")
        pl = QGridLayout(self.grp_params)
        pl.addWidget(QLabel("Количество кластеров (пусто = авто):"), 0, 0)
        self.spn_k = QSpinBox()
        self.spn_k.setRange(0, 10_000)
        self.spn_k.setValue(0)  # 0 = авто
        pl.addWidget(self.spn_k, 0, 1)

        h.addWidget(self.grp_params)  # добавляем в горизонтальный layout

        # ----------------- Добавляем горизонтальный layout в основной -----------------
        v.addLayout(h)

        # Кнопка запуска
        self.btn_run = QPushButton("Визуализировать кластеры")
        self.btn_run.clicked.connect(self.on_run_clicked)
        # Светло-зеленый цвет фона, черный текст
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #ADD8E6;  /* светло-зеленый */
                color: black;
                font-family: "Arial";   /* укажите ваш шрифт */
                font-size: 12pt;        /* размер шрифта */
            }
            QPushButton:hover {
                background-color: #90EE90;  /* чуть темнее при наведении */
            }
        """)

        v.addWidget(self.btn_run)

        # Метрики
        self.btn_results = QPushButton("Просмотр результатов")
        self.btn_results.clicked.connect(self.results_window.show)
        v.addWidget(self.btn_results)

        # Действия
        actions = QHBoxLayout()
        self.btn_load_ref = QPushButton("Загрузить эталонные данные")
        self.btn_load_ref.clicked.connect(self.on_load_reference)
        self.btn_clear_ref = QPushButton("Очистить эталон")
        self.btn_clear_ref.clicked.connect(self.on_clear_reference)
        self.btn_export = QPushButton("Сохранить графики")
        self.btn_export.clicked.connect(self.on_export_results)
        actions.addWidget(self.btn_load_ref)
        actions.addWidget(self.btn_clear_ref)
        actions.addWidget(self.btn_export)
        v.addLayout(actions)

        v.addStretch(1)

    # ---------- Слот: загрузка данных ----------
    def on_load_clicked(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл данных", "",
            "Текстовые/CSV (*.txt *.csv);;Все файлы (*.*)" )
        if not fname:
            return
        ok, message = self.data_processor.load_data(fname)
        self.lbl_filename.setText(Path(fname).name)
        if ok:
            self.lbl_status.setStyleSheet("color: green;")
            self.lbl_status.setText(message)
            self._populate_feature_checks()
            self._populate_data_table()
        else:
            self.lbl_status.setStyleSheet("color: red;")
            self.lbl_status.setText(message)
            QMessageBox.critical(self, "Ошибка", "Формат файла должен быть txt")

    def _populate_feature_checks(self):
        # очистка
        for i in reversed(range(self.feat_layout.count())):
            item = self.feat_layout.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.feature_checks.clear()

        # создание чекбоксов
        for i in range(self.data_processor.num_features):
            cb = QCheckBox(f"Признак {i+1}")
            cb.setChecked(True)
            self.feature_checks.append(cb)
            self.feat_layout.insertWidget(self.feat_layout.count() - 1, cb)

    def _populate_data_table(self, has_labels=False):
        self.tbl_data.clear()
        if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
            return
        data = self.data_processor.data
        if len(data) == 0:
            return
        num_features = getattr(self.data_processor, 'num_features', len(data[0]))
        self.tbl_data.setRowCount(len(data))
        self.tbl_data.setColumnCount(num_features)
        self.tbl_data.setHorizontalHeaderLabels([f"Признак {i+1}" for i in range(num_features)])
        for r, row in enumerate(data):
            for c in range(num_features):
                self.tbl_data.setItem(r, c, QTableWidgetItem(str(row[c])))
        self.tbl_data.resizeColumnsToContents()

    # ---------- Запуск кластеризации ----------
    def on_run_clicked(self):
        selected = [i for i, cb in enumerate(self.feature_checks) if cb.isChecked()]
        if not selected:
            QMessageBox.critical(self, "Ошибка", "Выберите хотя бы один признак")
            return

        # Подготовка данных
        data = self.data_processor.get_selected_data(selected)
        if data is None or len(data) == 0:
            QMessageBox.critical(self, "Ошибка", "Нет данных для кластеризации")
            return

        n_clusters = self.spn_k.value() or None
        method = self.cmb_method.currentText()
        num_classes = getattr(self.data_processor, 'num_classes', None)

        # Диалог загрузки
        self.loading = LoadingDialog(self, text="Идёт кластеризация…")

        # Поток
        self.worker = ClusteringWorker(data, method, n_clusters, num_classes, self.metrics_calculator)
        self.worker.finished_ok.connect(self._on_clustering_ok)
        self.worker.failed.connect(self._on_clustering_failed)

        # Запускаем: сначала показываем модалку, затем стартуем поток
        self.loading.show()
        self.worker.start()

    @QtCore.Slot(np.ndarray, dict)
    def _on_clustering_ok(self, labels, metrics):
        # Закрываем загрузку
        if hasattr(self, 'loading') and self.loading.isVisible():
            self.loading.close()

        # Получаем выбранные данные
        selected = [i for i, cb in enumerate(self.feature_checks) if cb.isChecked()]
        data = self.data_processor.get_selected_data(selected)
        self.cluster_labels = labels
        self.data = data
        metrics_result = dict(metrics)

            # Визуализация через Matplotlib widget
        algo_name = self.cmb_method.currentText()
        try:
            fig_widget = self.visualizer.get_matplotlib_widget(data, labels, title=f"Кластеры — {algo_name}")

            if not hasattr(self, 'results_window'):
                self.results_window = ResultsWindow()

            # Передаем данные и метки для возможности экспорта
            self.results_window.add_result(
                fig_widget,
                metrics_result,
                title=f"Результаты — {algo_name}",
                data=data,
                cluster_labels=labels
            )
            self.results_window.show()
            self.results_window.raise_()
            self.results_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при построении графика: {e}")

    @QtCore.Slot(str)
    def _on_clustering_failed(self, err):
        if hasattr(self, 'loading') and self.loading.isVisible():
            self.loading.close()
        QMessageBox.critical(self, "Ошибка", f"Ошибка при кластеризации: {err}")


    # ---------- Эталонные данные ----------
    def on_load_reference(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с эталонными данными", "",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )
        if not fname:
            return

        try:
            with open(fname, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) != 3:
                    QMessageBox.critical(self, "Ошибка", "Первая строка должна содержать 3 числа")
                    return

                num_features = int(parts[0])
                num_samples = int(parts[1])
                num_classes = int(parts[2])

                if num_samples != self.data_processor.num_samples:
                    QMessageBox.critical(
                        self, "Ошибка",
                        f"Образцов в эталоне ({num_samples}) ≠ загружено ({self.data_processor.num_samples})"
                    )
                    return

                data = []
                labels = []
                for line in f:
                    row = list(map(float, line.strip().split()))
                    if len(row) != num_features + 1:
                        QMessageBox.critical(
                            self, "Ошибка",
                            f"Ожидалось {num_features + 1} значений в строке"
                        )
                        return
                    data.append(row[:-1])
                    labels.append(int(row[-1]))

            self.data_processor.reference_data = np.array(data)
            self.data_processor.reference_labels = np.array(labels)
            self.data_processor.num_classes = num_classes

            self._populate_ref_table()

            # --- Уведомление об успешной загрузке ---
            QMessageBox.information(self, "Успех", f"Эталон загружен ({num_classes} классов)")

            # --- Визуализация и метрики ---
            try:
                vis = Visualizer()
                fig_widget = vis.get_matplotlib_widget(
                    np.array(data),
                    np.array(labels),
                    title="Эталонные кластеры"
                )

                # Рассчёт метрик эталона, если есть metrics_calculator
                metrics_result = {}
                if hasattr(self, 'metrics_calculator') and self.metrics_calculator is not None:
                    try:
                        metrics_result = self.metrics_calculator.calculate_all(
                            np.array(data), np.array(labels)
                        )
                    except Exception as metric_err:
                        QMessageBox.warning(self, "Метрики", f"Не удалось рассчитать метрики эталона: {metric_err}")

                # Создаем окно результатов, если его нет
                if not hasattr(self, 'results_window'):
                    self.results_window = ResultsWindow()

                # Добавляем результат в окно
                self.results_window.add_result(fig_widget, metrics=metrics_result, title="Эталонные кластеры")

                # Показываем окно
                self.results_window.show()
                self.results_window.raise_()
                self.results_window.activateWindow()

            except Exception as plot_err:
                QMessageBox.warning(self, "Визуализация", f"Не удалось построить график: {plot_err}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить эталон: {e}")

    def on_clear_reference(self):
        self.data_processor.reference_labels = None
        self.tbl_ref.clear()
        self.tbl_ref.setRowCount(0)
        self.tbl_ref.setColumnCount(0)
        QMessageBox.information(self, "Готово", "Эталонные данные очищены")

    def _populate_ref_table(self):
        self.tbl_ref.clear()
        if not (hasattr(self.data_processor, 'reference_data') and self.data_processor.reference_data is not None and
                hasattr(self.data_processor, 'reference_labels') and self.data_processor.reference_labels is not None):
            return
        ref_data = self.data_processor.reference_data
        ref_labels = self.data_processor.reference_labels
        rows = len(ref_labels)
        cols = ref_data.shape[1]
        self.tbl_ref.setRowCount(rows)
        self.tbl_ref.setColumnCount(cols + 1)
        self.tbl_ref.setHorizontalHeaderLabels([f"Признак {i+1}" for i in range(cols)] + ["Метка"])
        for r in range(rows):
            for c in range(cols):
                self.tbl_ref.setItem(r, c, QTableWidgetItem(str(ref_data[r, c])))
            self.tbl_ref.setItem(r, cols, QTableWidgetItem(str(int(ref_labels[r]))))
        self.tbl_ref.resizeColumnsToContents()

    # ---------- Экспорт ----------
    def on_export_results(self):
        if not hasattr(self, 'results_window'):
            QMessageBox.warning(self, "Экспорт", "Окно результатов не открыто")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты как картинку", "", "PNG (*.png)"
        )
        if not fname:
            return

        try:
            # Получаем контент внутри scroll_area
            content = self.results_window.scroll_content
            # Создаём pixmap нужного размера
            pixmap = QPixmap(content.size())
            # Рисуем контент на pixmap
            content.render(pixmap)
            # Сохраняем
            pixmap.save(fname)
            QMessageBox.information(self, "Экспорт", f"Результаты сохранены в {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение: {e}")


def main():
    app = QApplication(sys.argv)

    # Принудительно применяем стиль Fusion, который учитывает палитру
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(255, 255, 255))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 225))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(30, 144, 255))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    app.setPalette(palette)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
