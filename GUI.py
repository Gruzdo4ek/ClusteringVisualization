import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from DataManager import DataProcessor
from ClusterEngine import KMeansClustering
from Visualization import Visualizer
from Metrics import MetricsCalculator
from DBHandler import Database


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ кластеризации данных")
        self.root.geometry("700x750")  # Уменьшил размер окна

        self.style = ttk.Style()
        self.configure_styles()

        # Главный контейнер с прокруткой
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Canvas и скроллбар
        self.canvas = tk.Canvas(self.main_container)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Настройка прокрутки
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Инициализация компонентов
        self.data_processor = DataProcessor()
        self.clustering = KMeansClustering()
        self.visualizer = Visualizer()
        self.metrics_calculator = MetricsCalculator()
        self.database = Database()
        self.feature_vars = []  # Для хранения переменных чекбоксов

        self.create_widgets()
        self.setup_layout()

    def configure_styles(self):
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 9))  # Уменьшил шрифт
        self.style.configure('TButton', font=('Helvetica', 9), padding=3)  # Уменьшил шрифт и отступы
        self.style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))  # Уменьшил шрифт
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Error.TLabel', foreground='red')
        self.style.map('TButton',
                       foreground=[('active', '!disabled', 'black')],
                       background=[('active', '#4a7abc')])

    def create_widgets(self):
        # Заголовок
        self.header = ttk.Frame(self.scrollable_frame)
        self.title_label = ttk.Label(
            self.header,
            text="Анализ кластеризации методом k-средних",
            style='Header.TLabel'
        )

        # Секция загрузки данных
        self.load_frame = ttk.LabelFrame(self.scrollable_frame, text="Загрузка данных")
        self.load_button = ttk.Button(
            self.load_frame,
            text="Выбрать файл",
            command=self.load_file
        )
        self.file_label = ttk.Label(self.load_frame, text="Файл не выбран")
        self.status_label = ttk.Label(self.load_frame, text="", style='Success.TLabel')

        # Секция выбора признаков (теперь с чекбоксами)
        self.features_frame = ttk.LabelFrame(self.scrollable_frame, text="Выбор признаков")
        self.features_container = ttk.Frame(self.features_frame)
        self.features_canvas = tk.Canvas(self.features_container, height=50)  # Фиксированная высота
        self.features_inner_frame = ttk.Frame(self.features_canvas)

        # Настройка прокрутки для контейнера с чекбоксами
        self.features_inner_frame.bind(
            "<Configure>",
            lambda e: self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
        )

        self.features_canvas.create_window((0, 0), window=self.features_inner_frame, anchor="nw")

        # Параметры кластеризации
        self.params_frame = ttk.LabelFrame(self.scrollable_frame, text="Параметры кластеризации")
        self.clusters_label = ttk.Label(
            self.params_frame,
            text="Количество кластеров (оставьте пустым для автоматического выбора):"
        )
        self.clusters_entry = ttk.Entry(self.params_frame, width=10)

        # Таблицы данных
        self.tables_frame = ttk.Frame(self.scrollable_frame)

        # Таблица загруженных данных
        self.data_table_frame = ttk.LabelFrame(self.tables_frame, text="Загруженные данные")
        self.data_table = ttk.Treeview(
            self.data_table_frame,
            show="headings",
            height=6  # Уменьшил высоту таблицы
        )
        self.data_table_scroll = ttk.Scrollbar(
            self.data_table_frame,
            orient="vertical",
            command=self.data_table.yview
        )
        self.data_table.configure(yscrollcommand=self.data_table_scroll.set)

        # Таблица эталонных данных
        self.ref_table_frame = ttk.LabelFrame(self.tables_frame, text="Эталонные данные")
        self.ref_table = ttk.Treeview(
            self.ref_table_frame,
            show="headings",
            height=6  # Уменьшил высоту таблицы
        )
        self.ref_table_scroll = ttk.Scrollbar(
            self.ref_table_frame,
            orient="vertical",
            command=self.ref_table.yview
        )
        self.ref_table.configure(yscrollcommand=self.ref_table_scroll.set)

        # Метрики
        self.metrics_frame = ttk.LabelFrame(self.scrollable_frame, text="Метрики качества")
        self.metrics_text = ScrolledText(
            self.metrics_frame,
            height=8,  # Уменьшил высоту текстового поля
            wrap=tk.WORD,
            font=('Consolas', 9)  # Уменьшил шрифт
        )

        # Панель действий
        self.actions_frame = ttk.Frame(self.scrollable_frame)
        self.ref_button = ttk.Button(
            self.actions_frame,
            text="Загрузить эталонные данные",
            command=self.load_reference
        )
        self.clear_ref_button = ttk.Button(
            self.actions_frame,
            text="Очистить таблицу с эталонными данными",
            command=self.clear_reference_data,
            state=tk.DISABLED
        )
        self.export_button = ttk.Button(
            self.actions_frame,
            text="Сохранить результаты",
            command=self.export_results
        )

        # Кнопка запуска
        self.run_button = ttk.Button(
            self.scrollable_frame,
            text="Визуализировать кластеры",
            command=self.run_clustering
        )

    def setup_layout(self):
        # Заголовок
        self.header.pack(pady=5, fill=tk.X)  # Уменьшил отступ
        self.title_label.pack()

        # Секция загрузки данных
        self.load_frame.pack(pady=3, padx=5, fill=tk.X)  # Уменьшил отступы
        self.load_button.grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)  # Уменьшил отступы
        self.file_label.grid(row=0, column=1, padx=3, sticky=tk.W)  # Уменьшил отступы
        self.status_label.grid(row=1, column=0, columnspan=2, pady=3, sticky=tk.W)  # Уменьшил отступы

        # Таблицы данных (расположены горизонтально)
        self.tables_frame.pack(pady=3, padx=5, fill=tk.BOTH, expand=True)  # Уменьшил отступы

        # Таблица загруженных данных
        self.data_table_frame.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)  # Уменьшил отступы
        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.data_table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Таблица эталонных данных
        self.ref_table_frame.pack(side=tk.LEFT, padx=3, fill=tk.BOTH, expand=True)  # Уменьшил отступы
        self.ref_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ref_table_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Секция выбора признаков (с чекбоксами)
        self.features_frame.pack(pady=3, padx=5, fill=tk.X)  # Уменьшил отступы
        self.features_container.pack(fill=tk.BOTH, expand=True)
        self.features_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        # Параметры кластеризации
        self.params_frame.pack(pady=3, padx=5, fill=tk.X)  # Уменьшил отступы
        self.clusters_label.grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)  # Уменьшил отступы
        self.clusters_entry.grid(row=0, column=1, padx=3, pady=3, sticky=tk.W)  # Уменьшил отступы


        # Кнопка запуска
        self.run_button.pack(pady=5)  # Уменьшил отступ

        # Метрики
        self.metrics_frame.pack(pady=3, padx=5, fill=tk.BOTH, expand=True)  # Уменьшил отступы
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Панель действий
        self.actions_frame.pack(pady=5, fill=tk.X)  # Уменьшил отступ
        self.ref_button.pack(side=tk.LEFT, padx=3)  # Уменьшил отступы
        self.clear_ref_button.pack(side=tk.LEFT, padx=3)  # Уменьшил отступы
        self.export_button.pack(side=tk.LEFT, padx=3)  # Уменьшил отступы

    # Остальные методы остаются без изменений
    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[("Текстовые файлы", "*.txt"), ("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )

        if filepath:
            self.file_label.config(text=filepath.split('/')[-1])
            success, message = self.data_processor.load_data(filepath)

            if success:
                self.status_label.config(text=message, style='Success.TLabel')
                self.update_features_checkboxes()
                self.update_data_table()

            else:
                self.status_label.config(text=message, style='Error.TLabel')
                messagebox.showerror("Ошибка",  "Формат файла должен быть txt")

    def update_features_checkboxes(self):
        # Очищаем предыдущие чекбоксы
        for widget in self.features_inner_frame.winfo_children():
            widget.destroy()

        self.feature_vars = []

        # Создаем чекбоксы для каждого признака
        for i in range(self.data_processor.num_features):
            var = tk.BooleanVar(value=True)  # Все чекбоксы выбраны по умолчанию
            self.feature_vars.append(var)
            cb = ttk.Checkbutton(
                self.features_inner_frame,
                text=f"Признак {i + 1}",
                variable=var,
                onvalue=True,
                offvalue=False
            )
            cb.pack(anchor=tk.W)

    def run_clustering(self):
        # Получаем индексы выбранных признаков
        selected_indices = [i for i, var in enumerate(self.feature_vars) if var.get()]

        if not selected_indices:
            messagebox.showerror("Ошибка", "Выберите хотя бы один признак")
            return

        n_clusters = self.clusters_entry.get()
        try:
            n_clusters = int(n_clusters) if n_clusters else None
        except ValueError:
            messagebox.showerror("Ошибка", "Количество кластеров должно быть целым числом")
            return

        data = self.data_processor.get_selected_data(selected_indices)

        try:
            # Если есть эталонные метки, показываем их ДО кластеризации
            if hasattr(self.data_processor, 'reference_labels') and self.data_processor.reference_labels is not None:
                self.visualizer.plot_reference_clusters(data, self.data_processor.reference_labels)

            # Запускаем кластеризацию и показываем результат
            num_classes = getattr(self.data_processor, 'num_classes', None)
            labels = self.clustering.fit_predict(data, n_clusters, num_classes)
            self.visualizer.plot_clusters(data, labels)

            metrics = self.metrics_calculator.calculate_all(data, labels)

            if hasattr(self.data_processor, 'reference_labels') and self.data_processor.reference_labels is not None:
                ref_metrics = self.metrics_calculator.calculate_comparative(
                    self.data_processor.reference_labels,
                    labels
                )
                metrics.update(ref_metrics)

            self.show_metrics(metrics)
            self.cluster_labels = labels  # сохраняем метки кластеров
            self.data = data  # сохраняем данные
            self.metrics_result = metrics  # сохраняем метрики для экспорта


        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при кластеризации: {str(e)}")

    def show_metrics(self, metrics):
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "# Результаты кластеризации\n\n")

        max_name_length = max(len(name) for name in metrics.keys())
        for name, value in metrics.items():
            self.metrics_text.insert(tk.END, f"| {name.ljust(max_name_length)} | {value:.6f} |\n")

        self.metrics_text.insert(tk.END, "|" + "-" * (max_name_length + 2) + "|" + "-" * 12 + "|\n")

    def load_reference(self):
        filepath = filedialog.askopenfilename(
            title="Выберите файл с эталонными данными",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as file:
                    first_line = file.readline().strip()
                    parts = first_line.split()

                    if len(parts) != 3:
                        messagebox.showerror("Ошибка", "Файл эталонных данных должен содержать 3 числа в первой строке")
                        return

                    num_features = int(parts[0])
                    num_samples = int(parts[1])
                    num_classes = int(parts[2])

                    if num_samples != self.data_processor.num_samples:
                        messagebox.showerror("Ошибка",
                                             f"Количество образцов в эталонных данных ({num_samples}) не совпадает с загруженными данными ({self.data_processor.num_samples})")
                        return

                    # Читаем все данные
                    data = []
                    labels = []
                    for line in file:
                        row = list(map(float, line.strip().split()))
                        if len(row) != num_features + 1:
                            messagebox.showerror("Ошибка",
                                                 f"Ожидается {num_features + 1} значений (признаки + метка), получено {len(row)}")
                            return
                        data.append(row[:-1])  # Все кроме последнего элемента - признаки
                        labels.append(int(row[-1]))  # Последний элемент - метка

                    # Сохраняем данные
                    self.data_processor.reference_data = np.array(data)
                    self.data_processor.reference_labels = np.array(labels)
                    self.data_processor.num_classes = num_classes

                    # Обновляем таблицу и активируем кнопку очистки
                    self.update_ref_table()
                    self.clear_ref_button.config(state=tk.NORMAL)
                    messagebox.showinfo("Успех",
                                        f"Эталонные данные успешно загружены ({num_classes} классов)")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить эталонные данные: {str(e)}")

    def save_results(self):
        try:
            messagebox.showinfo("Успех", "Результаты сохранены в базу данных")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")

    def clear_reference_data(self):
        self.data_processor.reference_labels = None
        self.clear_ref_button.config(state=tk.DISABLED)
        self.ref_table.delete(*self.ref_table.get_children())
        messagebox.showinfo("Успех", "Эталонные данные удалены")

    def update_data_table(self):
        """Обновляет таблицу с загруженными данными"""
        try:
            # Очищаем предыдущие данные
            self.data_table.delete(*self.data_table.get_children())

            # Проверяем наличие данных
            if not hasattr(self.data_processor, 'data') or self.data_processor.data is None:
                return

            # Проверяем, что данные не пустые
            if len(self.data_processor.data) == 0:
                return

            # Получаем количество признаков из самих данных, если num_features не установлено
            num_features = getattr(self.data_processor, 'num_features',
                                   len(self.data_processor.data[0]) if len(self.data_processor.data) > 0 else 0)

            # Настраиваем колонки
            columns = [f"Признак {i + 1}" for i in range(num_features)]
            self.data_table["columns"] = columns

            # Конфигурируем заголовки колонок
            for col in columns:
                self.data_table.heading(col, text=col)
                self.data_table.column(col, width=80, anchor='center')  # Центрируем данные

            # Заполняем таблицу данными с проверкой каждой строки
            for row in self.data_processor.data:
                # Проверяем, что строка имеет правильное количество элементов
                if len(row) != num_features:
                    print(f"Предупреждение: строка {row} имеет неверное количество признаков")
                    continue

                self.data_table.insert("", tk.END, values=row)

        except Exception as e:
            print(f"Ошибка при обновлении таблицы данных: {str(e)}")
            # Можно добавить всплывающее окно с ошибкой, если нужно
            # messagebox.showerror("Ошибка", f"Не удалось обновить таблицу: {str(e)}")

    def update_ref_table(self):
        # Очищаем предыдущие данные
        self.ref_table.delete(*self.ref_table.get_children())

        # Проверяем, есть ли эталонные данные
        if (hasattr(self.data_processor, 'reference_data') and
                self.data_processor.reference_data is not None and
                hasattr(self.data_processor, 'reference_labels') and
                self.data_processor.reference_labels is not None):

            # Настраиваем колонки
            columns = [f"Признак {i + 1}" for i in range(self.data_processor.reference_data.shape[1])] + ["Метка"]
            self.ref_table["columns"] = columns

            for col in columns:
                self.ref_table.heading(col, text=col)
                self.ref_table.column(col, width=80)  # Уменьшил ширину колонок

            # Заполняем данными
            for i in range(len(self.data_processor.reference_labels)):
                row_data = list(self.data_processor.reference_data[i]) + [self.data_processor.reference_labels[i]]
                self.ref_table.insert("", tk.END, values=row_data)

    def export_results(self):
        try:
            messagebox.showinfo("Успех", "Результаты сохранены в базу данных")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")

        filepath = filedialog.asksaveasfilename(
            title="Экспорт результатов",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    # === Метрики ===
                    f.write("=== Метрики кластеризации ===\n")
                    if hasattr(self, 'metrics_result'):
                        for name, value in self.metrics_result.items():
                            f.write(f"{name}: {value}\n")
                    else:
                        f.write("Метрики отсутствуют.\n")

                    # === Данные + метки ===
                    f.write("\n=== Объекты и метки кластеров ===\n")
                    if hasattr(self, 'data') and hasattr(self, 'cluster_labels'):
                        import pandas as pd
                        df = pd.DataFrame(self.data)
                        df['cluster'] = self.cluster_labels
                        df.to_csv(f, sep='\t', index=False, header=False)
                    else:
                        f.write("Нет доступных данных или меток для экспорта.\n")

                messagebox.showinfo("Успех", "Результаты успешно экспортированы")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()