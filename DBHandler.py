
import sqlite3
from datetime import datetime

class Database:
    def __init__(self, db_name="clustering_system.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()

        # 1. Dataset
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Dataset (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                object_count INTEGER,
                feature_count INTEGER,
                upload_date DATETIME
            )
        """)

        # 2. ClusteringParams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ClusteringParams (
                params_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_count INTEGER,
                dataset_id INTEGER,
                FOREIGN KEY (dataset_id) REFERENCES Dataset(dataset_id)
            )
        """)

        # 3. ClusteringResult
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ClusteringResult (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                execution_date DATETIME,
                params_id INTEGER,
                silhouette_index REAL,
                davies_bouldin_index REAL,
                purity REAL,
                FOREIGN KEY (params_id) REFERENCES ClusteringParams(params_id)
            )
        """)

        # 4. ReferenceData
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ReferenceData (
                reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                object_count INTEGER,
                feature_count INTEGER,
                cluster_count INTEGER,
                upload_date DATETIME
            )
        """)

        self.conn.commit()

    # -------------------- INSERT METHODS --------------------

    def insert_dataset(self, file_path, object_count, feature_count):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO Dataset (file_path, object_count, feature_count, upload_date)
            VALUES (?, ?, ?, ?)
        """, (file_path, object_count, feature_count, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid

    def insert_clustering_params(self, cluster_count, dataset_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ClusteringParams (cluster_count, dataset_id)
            VALUES (?, ?)
        """, (cluster_count, dataset_id))
        self.conn.commit()
        return cursor.lastrowid

    def insert_clustering_result(self, file_path, params_id, silhouette_index, davies_bouldin_index, purity=None):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ClusteringResult (
                file_path, execution_date, params_id, silhouette_index, davies_bouldin_index, purity
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            file_path,
            datetime.now(),
            params_id,
            silhouette_index,
            davies_bouldin_index,
            purity
        ))
        self.conn.commit()
        return cursor.lastrowid

    def insert_reference_data(self, file_path, object_count, feature_count, cluster_count):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ReferenceData (file_path, object_count, feature_count, cluster_count, upload_date)
            VALUES (?, ?, ?, ?, ?)
        """, (file_path, object_count, feature_count, cluster_count, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                num_clusters INTEGER,
                inertia REAL,
                silhouette REAL,
                davies_bouldin REAL
            )
        """)
        self.conn.commit()

    def save_result(self, metrics, num_clusters):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO results (timestamp, num_clusters, inertia, silhouette, davies_bouldin)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            num_clusters,
            metrics["Инерция"],
            metrics["Индекс силуэта"],
            metrics["Индекс Давида-Болдуина"]
        ))
        self.conn.commit()

    def get_latest_clustering_results(self, limit=1):
        """Получить последние результаты кластеризации (по дате выполнения)"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT * FROM ClusteringResult
               ORDER BY execution_date DESC
               LIMIT ?
           """, (limit,))
        return cursor.fetchall()

    def get_clustering_result_by_id(self, result_id):
        """Получить результат кластеризации по ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT * FROM ClusteringResult
               WHERE result_id = ?
           """, (result_id,))
        return cursor.fetchone()

    def get_clustering_metrics_by_result_id(self, result_id):
        """Получить метрики кластеризации по result_id"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT silhouette_index, davies_bouldin_index, purity
               FROM ClusteringResult
               WHERE result_id = ?
           """, (result_id,))
        return cursor.fetchone()

    def get_dataset_by_id(self, dataset_id):
        """Получить информацию о датасете по его ID"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT * FROM Dataset
               WHERE dataset_id = ?
           """, (dataset_id,))
        return cursor.fetchone()

    def get_params_by_result_id(self, result_id):
        """Получить параметры кластеризации по result_id"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT cp.*
               FROM ClusteringParams cp
               JOIN ClusteringResult cr ON cp.params_id = cr.params_id
               WHERE cr.result_id = ?
           """, (result_id,))
        return cursor.fetchone()

    def get_all_results_summary(self):
        """Получить сводную таблицу всех результатов"""
        cursor = self.conn.cursor()
        cursor.execute("""
               SELECT cr.result_id, cr.execution_date, cr.silhouette_index, cr.davies_bouldin_index, cr.purity,
                      cp.cluster_count, ds.file_path
               FROM ClusteringResult cr
               JOIN ClusteringParams cp ON cr.params_id = cp.params_id
               JOIN Dataset ds ON cp.dataset_id = ds.dataset_id
               ORDER BY cr.execution_date DESC
           """)
        return cursor.fetchall()