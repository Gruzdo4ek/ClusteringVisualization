import unittest
import numpy as np
from Metrics import MetricsCalculator


class InertiaTestCase(unittest.TestCase):
    def setUp(self):
        self.calc = MetricsCalculator()

    def test_simple_clusters(self):
        data = np.array([(0, 0), (0, 2)])
        labels = np.array([0, 0])
        inertia = self.calc.calculate_inertia(data, labels)
        print(f"test_simple_clusters: Inertia = {inertia}")
        self.assertAlmostEqual(inertia, 2.0)

    def test_multiple_clusters(self):
        data = np.array([(1, 1), (1, 3), (2, 2), (4, 2)])
        labels = np.array([0, 0, 1, 1])
        inertia = self.calc.calculate_inertia(data, labels)
        print(f"test_multiple_clusters: Inertia = {inertia}")
        self.assertAlmostEqual(inertia, 4.0)

    def test_single_point(self):
        data = np.array([(5, 5)])
        labels = np.array([0])
        inertia = self.calc.calculate_inertia(data, labels)
        print(f"test_single_point: Inertia = {inertia}")
        self.assertEqual(inertia, 0.0)

    def test_float_points(self):
        data = np.array([(1.5, 2.5), (2.5, 2.5)])
        labels = np.array([0, 0])
        inertia = self.calc.calculate_inertia(data, labels)
        print(f"test_float_points: Inertia = {inertia}")
        self.assertAlmostEqual(inertia, 0.5)

    def test_large_cluster(self):
        data = np.array([(x, 0) for x in range(100)])
        labels = np.zeros(100, dtype=int)
        expected = sum((x - 49.5) ** 2 for x in range(100))
        inertia = self.calc.calculate_inertia(data, labels)
        print(f"test_large_cluster: Inertia = {inertia}")
        self.assertAlmostEqual(inertia, expected)

if __name__ == "__main__":
    unittest.main()