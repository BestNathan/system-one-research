import unittest

from run_two_minute_world import physics_frame, run_worldline
from run_probabilistic_world import initial_state


class FakeBackend:
    model = "fake"
    reported_model = "fake"

    def decide_choice(self, state, question_id, question):
        keys = list(question["criteria"])
        probs = {k: 0.0 for k in keys}
        chosen = "no_event" if "no_event" in probs else keys[0]
        if question_id == "driver_action":
            chosen = "keep_speed"
        probs[chosen] = 1.0
        return probs, {"usage": {"input_tokens": 1, "output_tokens": 0}}, 1.0


class TwoMinuteWorldTest(unittest.TestCase):
    def test_physics_12fps_keeps_speed_for_keep_action(self):
        state = initial_state()
        start = state["ego"]["speed_mph"]
        for _ in range(12):
            physics_frame(state, "keep_speed", dt=1 / 12)
        self.assertAlmostEqual(state["ego"]["speed_mph"], start, places=6)

    def test_short_run_has_expected_frame_count(self):
        result = run_worldline(
            FakeBackend(),
            0,
            duration_seconds=3,
            physics_fps=12,
            seed_base=20260930,
        )
        self.assertEqual(len(result["decisions"]), 3)
        self.assertEqual(len(result["frames"]), 36)
        self.assertEqual(result["decisions"][0]["frame_start"], 0)
        self.assertEqual(result["decisions"][0]["frame_end"], 11)
        self.assertEqual(result["decisions"][2]["frame_start"], 24)
        self.assertEqual(result["decisions"][2]["frame_end"], 35)


if __name__ == "__main__":
    unittest.main()
