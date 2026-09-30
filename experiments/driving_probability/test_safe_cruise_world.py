import unittest

from run_safe_cruise_world import (
    ACTION_ACCEL_MPS2,
    actor_question,
    apply_event,
    apply_speed_envelope,
    event_question,
    hard_brake_event_feasible,
    initial_state,
    physics_frame,
    refresh_control_state,
    safe_action_space,
)


class SafeCruiseWorldTest(unittest.TestCase):
    def make_state(self):
        return initial_state(
            target_speed_mph=80.0,
            min_gap_m=30.0,
            physics_fps=12,
        )

    def test_initial_state_starts_at_target_speed(self):
        state = self.make_state()
        self.assertEqual(state["ego"]["speed_mph"], 80.0)
        self.assertEqual(state["cruise"]["min_gap_m"], 30.0)
        self.assertEqual(state["cruise"]["target_gap_m"], 50.0)
        front = state["road"]["front_vehicle"]
        self.assertIsNotNone(front)
        self.assertEqual(front["distance_m"], 60.0)
        self.assertEqual(front["speed_mph"], 72.0)
        self.assertEqual(front["desired_speed_mph"], 76.0)

    def test_cut_in_respects_min_gap_with_margin(self):
        state = self.make_state()
        apply_event(state, "vehicle_cut_in")
        front = state["road"]["front_vehicle"]
        self.assertIsNotNone(front)
        self.assertGreaterEqual(front["distance_m"], 55.0)

    def test_action_filter_removes_unsafe_acceleration(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = {
            "distance_m": 34.0,
            "speed_mph": 60.0,
            "desired_speed_mph": 60.0,
        }
        refresh_control_state(state, 12)
        actions, predicted = safe_action_space(state, 12)
        self.assertNotIn("accelerate", actions)
        self.assertNotIn("hard_accelerate", actions)
        self.assertLess(predicted["accelerate"], 30.0)

    def test_low_speed_masks_braking(self):
        state = self.make_state()
        state["ego"]["speed_mph"] = 0.0
        actions, _ = safe_action_space(state, 12)
        self.assertNotIn("brake", actions)
        self.assertNotIn("hard_brake", actions)

    def test_barrier_shield_never_crosses_min_gap(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = {
            "distance_m": 30.2,
            "speed_mph": 50.0,
            "desired_speed_mph": 50.0,
        }
        speed, gap, margin, overridden, accel = physics_frame(
            state,
            "hard_accelerate",
            dt=1 / 12,
        )
        self.assertTrue(overridden)
        self.assertGreaterEqual(gap, 30.0)
        self.assertGreaterEqual(margin, 0.0)
        self.assertLess(accel, ACTION_ACCEL_MPS2["hard_accelerate"])

    def test_hard_brake_event_hidden_when_margin_is_too_small(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = {
            "distance_m": 35.0,
            "speed_mph": 72.0,
            "desired_speed_mph": 72.0,
        }
        self.assertFalse(hard_brake_event_feasible(state, 12))
        keys = set(event_question(state, 12)["criteria"])
        self.assertNotIn("lead_vehicle_hard_brake", keys)

    def test_overspeed_clear_road_forces_negative_actions(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = None
        state["ego"]["speed_mph"] = 90.0
        refresh_control_state(state, 12)
        actions = apply_speed_envelope(
            state,
            ["hard_brake", "brake", "coast", "keep_speed", "accelerate"],
        )
        self.assertTrue(actions)
        self.assertTrue(all(ACTION_ACCEL_MPS2[a] < 0 for a in actions))

    def test_underspeed_clear_road_forces_positive_actions(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = None
        state["ego"]["speed_mph"] = 60.0
        refresh_control_state(state, 12)
        actions = apply_speed_envelope(
            state,
            ["brake", "coast", "keep_speed", "accelerate", "hard_accelerate"],
        )
        self.assertEqual(actions, ["accelerate", "hard_accelerate"])

    def test_cruise_band_does_not_allow_one_second_overshoot(self):
        state = self.make_state()
        state["road"]["front_vehicle"] = None
        state["ego"]["speed_mph"] = 80.0
        refresh_control_state(state, 12)
        actions = apply_speed_envelope(
            state,
            ["coast", "keep_speed", "accelerate", "hard_accelerate"],
        )
        self.assertIn("keep_speed", actions)
        self.assertIn("coast", actions)
        self.assertNotIn("accelerate", actions)
        self.assertNotIn("hard_accelerate", actions)

    def test_actor_question_only_contains_safe_actions(self):
        state = self.make_state()
        actions = ["hard_brake", "brake", "coast"]
        q = actor_question(state, actions)
        self.assertEqual(list(q["criteria"]), actions)


if __name__ == "__main__":
    unittest.main()
