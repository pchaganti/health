import json
import os
import tempfile
import unittest
from unittest.mock import patch

from healthai.preferences import load_preferences, save_preferences


class PreferencesTests(unittest.TestCase):
    def test_legacy_export_path_keys_are_migrated_bidirectionally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_path = os.path.join(temp_dir, "prefs.json")
            with open(prefs_path, "w", encoding="utf-8") as handle:
                json.dump({"export_xml_path": "/tmp/export.xml"}, handle)

            with patch.dict(os.environ, {"APPLEHEALTH_PREFS": prefs_path}):
                preferences = load_preferences()
                save_preferences(preferences)

            self.assertEqual(preferences["export_xml"], "/tmp/export.xml")
            with open(prefs_path, encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["export_xml_path"], "/tmp/export.xml")
            self.assertEqual(persisted["export_xml"], "/tmp/export.xml")


if __name__ == "__main__":
    unittest.main()
