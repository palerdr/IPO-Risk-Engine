"""Check audit failure handling and report output isolation."""

import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ipo_research.expanded.report import report
from ipo_research.expanded.storage import save_json

ROOT = Path(__file__).resolve().parents[1]


class IntegrityTests(unittest.TestCase):
    def test_optimized_python_rejects_a_mismatched_dataset_hash(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("dataset", "input", "universe"):
                save_json(directory / f"{name}.json.gz", {})
            save_json(directory / "protocol.json", {})
            save_json(
                directory / "results.json.gz",
                {"provenance": {"dataset_sha256": "invalid"}},
            )
            process = subprocess.run(
                [
                    sys.executable,
                    "-O",
                    "-c",
                    "from pathlib import Path; import sys; "
                    "from ipo_research.expanded.verify import verify; "
                    "verify(Path(sys.argv[1]), Path(sys.argv[1]) / 'models')",
                    str(directory),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("ValueError: Audit mismatch:", process.stderr)
            self.assertIn("dataset_sha256", process.stderr)

    def test_report_uses_the_requested_output_and_keeps_markdown_tables_intact(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "research/expanded"
            directory.mkdir(parents=True)
            for name in ("dataset", "input", "universe", "results", "coverage-results"):
                shutil.copyfile(
                    ROOT / f"research/expanded/{name}.json.gz", directory / f"{name}.json.gz"
                )
            destination = root / "custom/reports"
            artifact = report(directory, destination)
            markdown = (destination / "CHALLENGER_STUDY.md").read_text()
            self.assertFalse((root / "docs").exists())
            self.assertEqual(artifact["manifest"]["title"], "IPO challenger study")
            self.assertIn(
                "| Stage | Model | Brier | Skill vs train rate | Skill vs 50% |", markdown
            )
            self.assertNotIn("\n\n\n", markdown)
            self.assertNotIn("|\n\n|", markdown)
            self.assertIn("final two blocks", markdown)
            self.assertEqual(len(artifact["manifest"]["charts"]), 3)
            self.assertEqual(len(artifact["manifest"]["tables"]), 9)


if __name__ == "__main__":
    unittest.main()
