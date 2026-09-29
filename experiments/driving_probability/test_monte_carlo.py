import unittest

from run_monte_carlo import aggregate, percentile


class MonteCarloMetricsTest(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.9), 5)

    def test_aggregate_sampling_frequency(self):
        actions = [
            "hard_brake",
            "brake",
            "keep_speed",
            "accelerate",
            "hard_accelerate",
        ]
        probs = {a: 0.2 for a in actions}
        worldlines = []
        for i, action in enumerate(actions):
            worldlines.append(
                {
                    "worldline_id": i,
                    "reported_model": "fake",
                    "steps": [
                        {
                            "step": 0,
                            "probabilities": probs,
                            "sampled_action": action,
                            "speed_after_mph": 20.0 + i,
                            "front_gap_after_m": 10.0,
                            "elapsed_ms": 1.0,
                        }
                    ],
                    "summary": {
                        "final_speed_mph": 20.0 + i,
                        "max_speed_mph": 20.0 + i,
                        "min_front_gap_m": 10.0,
                        "near_miss": False,
                        "critical_gap": False,
                        "collision_floor_hit": False,
                        "hard_brake_count": int(action == "hard_brake"),
                        "hard_accelerate_count": int(action == "hard_accelerate"),
                        "non_argmax_count": 0,
                        "input_tokens": 100,
                        "output_tokens": 10,
                    },
                }
            )

        result = aggregate(
            worldlines,
            backend="fake",
            requested_model="fake",
        )
        self.assertEqual(result["calls"], 5)
        self.assertEqual(result["usage"]["input_tokens"], 500)
        self.assertAlmostEqual(
            result["sampling_calibration"]["max_abs_frequency_minus_probability"],
            0.0,
        )
        self.assertAlmostEqual(result["outcomes"]["near_miss_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
