import tempfile
import unittest
from pathlib import Path

import pandas as pd

from healthai.chat import _build_context


class ChatContextTests(unittest.TestCase):
    def test_context_uses_complete_recent_rows_and_weekly_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dates = pd.date_range("2017-01-01", "2025-12-31", freq="D")
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "value": range(1, len(dates) + 1),
                }
            ).to_csv(output_dir / "steps_data.csv", index=False)

            context = _build_context(
                str(output_dir),
                "What were my most active weeks in 2025?",
            )

            self.assertIn("2025-12-31", context)
            self.assertIn("Weekly step totals", context)
            self.assertIn("2025", context)
            self.assertNotIn("2017-01-01", context)

    def test_context_does_not_end_with_a_partial_csv_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=500).strftime(
                        "%Y-%m-%d"
                    ),
                    "value": range(500),
                }
            ).to_csv(output_dir / "steps_data.csv", index=False)

            context = _build_context(str(output_dir), "Show my recent steps")

            self.assertTrue(context.endswith("\n"))
            self.assertIn("2026-05-15", context)

    def test_month_question_includes_totals_for_the_full_selected_year(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            dates = pd.date_range("2025-01-01", "2025-02-28", freq="D")
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "value": [100] * 31 + [200] * 28,
                }
            ).to_csv(output_dir / "steps_data.csv", index=False)

            context = _build_context(
                str(output_dir),
                "What were my most active months in 2025?",
            )

            self.assertIn("Monthly statistics", context)
            self.assertIn("2025-01-01,3100", context)
            self.assertIn("2025-02-01,5600", context)


if __name__ == "__main__":
    unittest.main()
