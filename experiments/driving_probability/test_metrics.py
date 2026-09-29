import unittest

from common import js_divergence, spearman, total_variation


class MetricsTest(unittest.TestCase):
    def test_js_zero_for_identical(self):
        p = {
            "hard_brake": 0.1,
            "brake": 0.2,
            "keep_speed": 0.4,
            "accelerate": 0.2,
            "hard_accelerate": 0.1,
        }
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=12)

    def test_total_variation(self):
        p = {
            "hard_brake": 1.0,
            "brake": 0.0,
            "keep_speed": 0.0,
            "accelerate": 0.0,
            "hard_accelerate": 0.0,
        }
        q = {
            "hard_brake": 0.0,
            "brake": 0.0,
            "keep_speed": 0.0,
            "accelerate": 0.0,
            "hard_accelerate": 1.0,
        }
        self.assertAlmostEqual(total_variation(p, q), 1.0)

    def test_spearman_monotone(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)


if __name__ == "__main__":
    unittest.main()
