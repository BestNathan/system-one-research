import unittest

from common import normalize_choice_probabilities
from run_probabilistic_world import (
    age_recent_history,
    apply_event,
    event_question,
    initial_state,
)


class ProbabilisticWorldTest(unittest.TestCase):
    def test_dynamic_event_space_without_lead(self):
        state = initial_state()
        keys = set(event_question(state)["criteria"])
        self.assertEqual(keys, {"no_event", "vehicle_cut_in"})

    def test_dynamic_event_space_with_lead(self):
        state = initial_state()
        apply_event(state, "vehicle_cut_in")
        keys = set(event_question(state)["criteria"])
        self.assertEqual(
            keys,
            {
                "no_event",
                "lead_vehicle_hard_brake",
                "lead_vehicle_accelerate",
                "lead_vehicle_exit",
            },
        )

    def test_history_expires(self):
        state = initial_state()
        state["recent_history"]["events"] = [
            {"kind": "vehicle_cut_in", "age_steps": 0}
        ]
        for _ in range(5):
            age_recent_history(state)
        self.assertEqual(state["recent_history"]["events"][0]["age_steps"], 5)
        age_recent_history(state)
        self.assertEqual(state["recent_history"]["events"], [])

    def test_generic_choice_normalization(self):
        answer = {
            "choice": "b",
            "probabilities": {"a": 2, "b": 3},
        }
        probs = normalize_choice_probabilities(answer, ["a", "b"])
        self.assertAlmostEqual(probs["a"], 0.4)
        self.assertAlmostEqual(probs["b"], 0.6)


if __name__ == "__main__":
    unittest.main()
