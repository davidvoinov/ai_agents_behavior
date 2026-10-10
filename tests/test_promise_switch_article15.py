from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Article15AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "article15"
        subprocess.run(
            [
                "python3", "-B", "scripts/analyze_promise_switch_final.py",
                "--profile", "article_15", "--output-dir", str(cls.output),
            ],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        cls.manifest = json.loads((cls.output / "analysis_manifest.json").read_text())
        with (cls.output / "confirmatory_h7_h9.csv").open(newline="") as handle:
            cls.confirmatory = list(csv.DictReader(handle))
        with (cls.output / "voluntary_itt.csv").open(newline="") as handle:
            cls.voluntary = list(csv.DictReader(handle))
        with (cls.output / "condition_counts.csv").open(newline="") as handle:
            cls.condition_counts = list(csv.DictReader(handle))
        with (cls.output / "promise_switch_variance_decomposition.csv").open(newline="") as handle:
            cls.variance = list(csv.DictReader(handle))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_configuration_set_order_and_exclusions(self):
        expected = [
            "gpt-oss-120b", "Qwen3-32B", "DeepSeek-V4-Flash",
            "DeepSeek-V4-Pro", "DeepSeek-V4-Pro (r)", "Kimi-K3",
            "MiniMax-M3", "MiniMax-M3 (r)", "GLM-5.2", "GLM-5.2 (r)",
            "Llama-3.3-70B", "GPT-5.6", "GPT-5.6 (r)", "Sonnet-5",
            "Sonnet-5 (r)",
        ]
        metadata = self.manifest["included_configurations"]
        self.assertEqual([row["configuration"] for row in metadata], expected)
        self.assertEqual(list(dict.fromkeys(row["configuration"] for row in self.condition_counts)), expected)
        self.assertEqual(list(dict.fromkeys(row["configuration"] for row in self.confirmatory)), expected)
        self.assertEqual([row["configuration"] for row in self.voluntary], expected)
        self.assertEqual(len(metadata), 15)
        self.assertFalse(any(row["configuration"] == "Qwen3-32B" and row["reasoning_mode"] == "on" for row in metadata))
        self.assertFalse(any(row["model_id"] == "openai/gpt-oss-120b" and row["provider"] == "groq" for row in metadata))

    def test_sample_counts_cells_and_duplicates(self):
        validation = self.manifest["validation"]["profile"]
        self.assertEqual(validation["configuration_count"], 15)
        self.assertEqual(validation["condition_count"], 6)
        self.assertEqual(validation["cell_count"], 90)
        self.assertEqual(validation["assigned_trials"], 9584)
        self.assertEqual(validation["valid_outcomes"], 9022)
        self.assertEqual(validation["missing_outcomes"], 562)
        self.assertEqual(validation["duplicate_trial_id_count"], 0)
        self.assertEqual(validation["duplicate_source_run_trial_id_count"], 0)
        self.assertEqual(validation["duplicate_exact_row_count"], 0)

    def test_hypotheses_and_holm_families(self):
        self.assertEqual({row["hypothesis"] for row in self.confirmatory}, {"H7", "H8", "H9"})
        self.assertTrue(all(int(row["holm_family_size"]) == 3 for row in self.confirmatory))
        self.assertEqual(len(self.voluntary), 15)
        self.assertTrue(all(int(row["holm_family_size"]) == 15 for row in self.voluntary))

    def test_variance_controls(self):
        values = {(row["model_type"], row["component"], row["metric"]): float(row["value"]) for row in self.variance}
        expected = {
            ("linear_probability_model", "configuration", "share_total_variance"): 0.289,
            ("linear_probability_model", "condition", "share_total_variance"): 0.173,
            ("linear_probability_model", "configuration_x_condition", "share_total_variance"): 0.163,
            ("linear_probability_model", "residual", "share_total_variance"): 0.374,
            ("linear_probability_model", "model", "r_squared"): 0.626,
            ("logistic_regression", "configuration", "share_explained_deviance"): 0.518,
            ("logistic_regression", "condition", "share_explained_deviance"): 0.339,
            ("logistic_regression", "configuration_x_condition", "share_explained_deviance"): 0.143,
            ("logistic_regression", "model", "mcfadden_pseudo_r_squared"): 0.591,
        }
        for key, target in expected.items():
            self.assertAlmostEqual(values[key], target, delta=0.0006)

    def test_public_semantic_input_and_portable_manifest(self):
        semantic = self.manifest["public_semantic_input"]
        self.assertEqual(semantic["row_count"], 3538)
        self.assertFalse(semantic["private_linkage_required"])
        self.assertIn("git_head_optional", self.manifest)
        self.assertNotIn("required_ancestor_commits", self.manifest)
        self.assertNotIn("private_inputs_used_but_not_redistributed", self.manifest)

    def test_default_output_inventory_contains_only_publication_figures(self):
        generated = set(self.manifest["generated_files"])
        self.assertTrue({
            "fig_PS_condition_heatmap.pdf", "fig_PS_condition_heatmap.png",
            "fig_PS_confirmatory_forest.pdf", "fig_PS_confirmatory_forest.png",
            "fig_PS_voluntary_itt.pdf", "fig_PS_voluntary_itt.png",
        }.issubset(generated))
        self.assertFalse(any(name.startswith("forest_") for name in generated))


if __name__ == "__main__":
    unittest.main()
