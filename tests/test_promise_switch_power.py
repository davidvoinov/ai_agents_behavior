from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from promise_switch.power import (
    alternative_rate,
    build_power_report,
    call_plan,
    initiated_trials,
    minimum_n_per_arm,
    two_proportion_z_power,
)

ROOT = Path(__file__).resolve().parents[1]


class PowerTests(unittest.TestCase):
    def test_power_increases_with_n(self) -> None:
        powers = [two_proportion_z_power(p0=.4, p1=.65, n_per_arm=n, alpha=.05)
                  for n in (50, 75, 100)]
        self.assertLess(powers[0], powers[1])
        self.assertLess(powers[1], powers[2])

    def test_power_increases_with_mde(self) -> None:
        powers = [two_proportion_z_power(p0=.4, p1=.4 + mde, n_per_arm=75, alpha=.05)
                  for mde in (.15, .20, .25, .30)]
        self.assertEqual(powers, sorted(powers))

    def test_lower_alpha_and_higher_target_require_at_least_as_much_n(self) -> None:
        nominal_80 = minimum_n_per_arm(p0=.4, p1=.65, alpha=.05, target_power=.8)
        conservative_80 = minimum_n_per_arm(p0=.4, p1=.65, alpha=.05 / 3, target_power=.8)
        nominal_90 = minimum_n_per_arm(p0=.4, p1=.65, alpha=.05, target_power=.9)
        self.assertGreaterEqual(conservative_80, nominal_80)
        self.assertGreaterEqual(nominal_90, nominal_80)

    def test_invalid_alternatives_rejected_and_grid_skipped(self) -> None:
        with self.assertRaisesRegex(ValueError, "p1"):
            alternative_rate(.9, .2, "increase")
        with self.assertRaisesRegex(ValueError, "p1"):
            alternative_rate(.1, .2, "decrease")
        report = build_power_report(
            reference_rates=(.9,), mdes=(.2,), candidate_n=(50,),
            alphas=(.05,), target_powers=(.8,),
        )
        self.assertEqual(len(report["candidate_power"]), 1)
        self.assertEqual(report["candidate_power"][0]["direction"], "decrease")

    def test_attrition_inflation(self) -> None:
        self.assertEqual(initiated_trials(50, 0), 50)
        self.assertEqual(initiated_trials(50, .02), 52)
        self.assertEqual(initiated_trials(75, .05), 79)
        self.assertEqual(initiated_trials(100, .10), 112)

    def test_exact_call_arithmetic(self) -> None:
        plan = call_plan(75, 8)
        self.assertEqual(plan["final_calls_per_configuration"], 450)
        self.assertEqual(plan["message_calls_per_configuration"], 150)
        self.assertEqual(plan["total_calls_per_configuration"], 600)
        self.assertEqual(plan["total_calls_all_configurations"], 4800)

    def test_matches_statsmodels_when_available(self) -> None:
        try:
            from statsmodels.stats.proportion import power_proportions_2indep
        except ImportError:
            self.skipTest("statsmodels is not installed")
        for p0, p1, n, alpha in ((.4, .65, 75, .05), (.6, .4, 100, .05 / 3), (.1, .4, 50, .05)):
            expected = power_proportions_2indep(
                p1 - p0, p0, n, ratio=1, alpha=alpha,
                alternative="two-sided", return_results=False,
            )
            actual = two_proportion_z_power(p0=p0, p1=p1, n_per_arm=n, alpha=alpha)
            self.assertAlmostEqual(actual, expected, places=12)
        expected_minimum = next(
            n for n in range(1, 1000)
            if power_proportions_2indep(
                .25, .4, n, ratio=1, alpha=.05,
                alternative="two-sided", return_results=False,
            ) >= .8
        )
        self.assertEqual(
            minimum_n_per_arm(p0=.4, p1=.65, alpha=.05, target_power=.8),
            expected_minimum,
        )

    def test_default_grid_and_cli_are_deterministic(self) -> None:
        report = build_power_report()
        self.assertEqual(len(report["candidate_power"]), 300)
        self.assertEqual(len(report["minimum_n"]), 200)
        command = [sys.executable, "-B", "scripts/power_promise_switch.py"]
        first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
        second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["method"]["fisher_exact_power"], False)


if __name__ == "__main__":
    unittest.main()
