import contextlib
import io
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

from healthai import cli
from healthai.chat import _build_context
from healthai.preferences import save_preferences


class CliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.export_path = self.root / "export.xml"
        health = ET.Element("HealthData")
        ET.SubElement(
            health,
            "Record",
            {
                "type": "HKQuantityTypeIdentifierStepCount",
                "sourceName": "Apple Watch",
                "unit": "count",
                "value": "1234",
                "startDate": "2026-07-01 09:00:00 +0800",
                "endDate": "2026-07-01 10:00:00 +0800",
            },
        )
        ET.ElementTree(health).write(
            self.export_path,
            encoding="utf-8",
            xml_declaration=True,
        )

    def test_csv_command_also_builds_chat_metric_files(self):
        cli._export_xml_path = str(self.export_path)
        cli._output_dir = str(self.root / "output")
        with contextlib.redirect_stdout(io.StringIO()):
            cli.convert_xml_to_csv()

        output = self.root / "output"
        self.assertTrue((output / "records.csv").exists())
        self.assertTrue((output / "steps_data.csv").exists())
        self.assertIn("1234", _build_context(str(output), "Show my recent steps"))

    def test_resolver_reads_setup_wizard_export_key(self):
        prefs_path = self.root / "prefs.json"
        with patch.dict(os.environ, {"APPLEHEALTH_PREFS": str(prefs_path)}):
            save_preferences({"export_xml_path": str(self.export_path)})
            cli._export_xml_path = None
            resolved = cli.resolve_export_xml()

        self.assertEqual(resolved, str(self.export_path))


if __name__ == "__main__":
    unittest.main()
