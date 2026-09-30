import unittest

from run_two_minute_world import (
    LOW_SPEED_BRAKE_MASK_MPH,
    MAX_RELEVANT_FRONT_GAP_M,
    RECENT_ACTION_MAX_AGE_STEPS,
    actor_question,
    age_recent_actions,
    initial_state_v2,
    physics_frame,
    prune_irrelevant_front_vehicle,
    run_worldline,
)


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
    def test_v2_state_removes_categorical_last_action(self):
        state = initial_state_v2()
        self.assertNotIn("last_action", state["recent_history"])
        self.assertEqual(state["recent_history"]["actions"], [])

    def test_low_speed_masks_braking_actions(self):
        state = initial_state_v2()
        state["ego"]["speed_mph"] = LOW_SPEED_BRAKE_MASK_MPH
        keys = set(actor_question(state)["criteria"])
        self.assertNotIn("brake", keys)
        self.assertNotIn("hard_brake", keys)
        self.assertIn("keep_speed", keys)
        self.assertIn("accelerate", keys)

    def test_normal_speed_keeps_full_action_space(self):
        state = initial_state_v2()
        state["ego"]["speed_mph"] = 30.0
        keys = set(actor_question(state)["criteria"])
        self.assertEqual(
            keys,
            {
                "hard_brake",
                "brake",
                "keep_speed",
                "accelerate",
                "hard_accelerate",
            },
        )

    def test_recent_action_history_expires(self):
        state = initial_state_v2()
        state["recent_history"]["actions"] = [
            {"longitudinal_acceleration_mps2": -2.0, "age_steps": 0}
        ]
        for _ in range(RECENT_ACTION_MAX_AGE_STEPS):
            age_recent_actions(state)
        self.assertEqual(
            state["recent_history"]["actions"][0]["age_steps"],
            RECENT_ACTION_MAX_AGE_STEPS,
        )
        age_recent_actions(state)
        self.assertEqual(state["recent_history"]["actions"], [])

    def test_irrelevant_front_vehicle_is_pruned(self):
        state = initial_state_v2()
        state["road"]["front_vehicle"] = {
            "distance_m": MAX_RELEVANT_FRONT_GAP_M + 0.1,
            "speed_mph": 40.0,
        }
        self.assertTrue(prune_irrelevant_front_vehicle(state))
        self.assertIsNone(state["road"]["front_vehicle"])

    def test_physics_12fps_keeps_speed_for_keep_action(self):
        state = initial_state_v2()
        start = state["ego"]["speed_mph"]
        for _ in range(12):
            physics_frame(state, "keep_speed", dt=1 / 12)
        self.assertAlmostEqual(state["ego"]["speed_mph"], start, places=6)

    def test_short_run_has_expected_frame_count_and_actor_options(self):
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
        self.assertIn("actor_options", result["decisions"][0])


if __name__ == "__main__":
    unittest.main()
