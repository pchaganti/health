import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from healthai.health_data import HealthDataSet


class HealthDataSetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.export_path = Path(self.temp_dir.name) / "export.xml"

    def write_export(self, records=(), workouts=()):
        root = ET.Element("HealthData")
        for attributes in records:
            ET.SubElement(root, "Record", attributes)
        for attributes, statistics in workouts:
            workout = ET.SubElement(root, "Workout", attributes)
            for statistic in statistics:
                ET.SubElement(workout, "WorkoutStatistics", statistic)
        ET.ElementTree(root).write(
            self.export_path,
            encoding="utf-8",
            xml_declaration=True,
        )
        return HealthDataSet(self.export_path)

    @staticmethod
    def quantity(
        record_type,
        value,
        unit,
        start,
        end,
        source="Apple Watch",
    ):
        return {
            "type": record_type,
            "sourceName": source,
            "unit": unit,
            "value": str(value),
            "startDate": start,
            "endDate": end,
        }

    def test_cumulative_samples_reconcile_overlapping_sources(self):
        record_type = "HKQuantityTypeIdentifierStepCount"
        dataset = self.write_export(
            records=[
                self.quantity(
                    record_type,
                    1000,
                    "count",
                    "2026-07-01 09:00:00 +0800",
                    "2026-07-01 10:00:00 +0800",
                    "iPhone",
                ),
                self.quantity(
                    record_type,
                    1000,
                    "count",
                    "2026-07-01 09:00:00 +0800",
                    "2026-07-01 10:00:00 +0800",
                    "Apple Watch",
                ),
            ]
        )

        daily = dataset.daily_quantity(record_type, "cumulative")

        self.assertEqual(daily.iloc[0], 1000)

    def test_source_priority_and_legacy_sum_mode_are_respected(self):
        record_type = "HKQuantityTypeIdentifierStepCount"
        dataset = self.write_export(
            records=[
                self.quantity(
                    record_type,
                    700,
                    "count",
                    "2026-07-01 09:00:00 +0800",
                    "2026-07-01 10:00:00 +0800",
                    "iPhone",
                ),
                self.quantity(
                    record_type,
                    1000,
                    "count",
                    "2026-07-01 09:00:00 +0800",
                    "2026-07-01 10:00:00 +0800",
                    "Apple Watch",
                ),
            ]
        )

        preferred = dataset.daily_quantity(
            record_type,
            "cumulative",
            source_priority=["iPhone"],
        )
        legacy = dataset.daily_quantity(
            record_type,
            "cumulative",
            source_mode="all",
        )

        self.assertEqual(preferred.iloc[0], 700)
        self.assertEqual(legacy.iloc[0], 1700)

    def test_distance_units_are_normalized_before_aggregation(self):
        record_type = "HKQuantityTypeIdentifierDistanceWalkingRunning"
        dataset = self.write_export(
            records=[
                self.quantity(
                    record_type,
                    1,
                    "km",
                    "2026-07-01 09:00:00 +0800",
                    "2026-07-01 10:00:00 +0800",
                ),
                self.quantity(
                    record_type,
                    1,
                    "mi",
                    "2026-07-01 10:00:00 +0800",
                    "2026-07-01 11:00:00 +0800",
                ),
            ]
        )

        daily = dataset.daily_quantity(record_type, "cumulative")

        self.assertAlmostEqual(daily.iloc[0], 2.609344, places=6)

    def test_mixed_timezone_offsets_do_not_break_daily_grouping(self):
        record_type = "HKQuantityTypeIdentifierStepCount"
        dataset = self.write_export(
            records=[
                self.quantity(
                    record_type,
                    1000,
                    "count",
                    "2026-01-01 09:00:00 -0800",
                    "2026-01-01 10:00:00 -0800",
                ),
                self.quantity(
                    record_type,
                    2000,
                    "count",
                    "2026-07-01 09:00:00 -0700",
                    "2026-07-01 10:00:00 -0700",
                ),
            ]
        )

        daily = dataset.daily_quantity(record_type, "cumulative")

        self.assertEqual(list(daily.values), [1000, 2000])

    def test_overlapping_sleep_sources_are_counted_once(self):
        records = [
            {
                "type": "HKCategoryTypeIdentifierSleepAnalysis",
                "sourceName": source,
                "value": "HKCategoryValueSleepAnalysisAsleepUnspecified",
                "startDate": "2026-07-01 23:00:00 +0800",
                "endDate": "2026-07-02 07:00:00 +0800",
            }
            for source in ("Apple Watch", "WHOOP")
        ]
        dataset = self.write_export(records=records)

        daily_sleep, _, _ = dataset.sleep_summary()

        self.assertEqual(daily_sleep.iloc[0], 8.0)
        self.assertEqual(str(daily_sleep.index[0]), "2026-07-02")

    def test_sleep_in_bed_and_stage_rows_are_not_added_together(self):
        sleep_type = "HKCategoryTypeIdentifierSleepAnalysis"
        common = {
            "type": sleep_type,
            "sourceName": "Apple Watch",
        }
        records = [
            {
                **common,
                "value": "HKCategoryValueSleepAnalysisInBed",
                "startDate": "2026-07-01 23:00:00 +0800",
                "endDate": "2026-07-02 07:00:00 +0800",
            },
            {
                **common,
                "value": "HKCategoryValueSleepAnalysisAsleepCore",
                "startDate": "2026-07-01 23:00:00 +0800",
                "endDate": "2026-07-02 03:00:00 +0800",
            },
            {
                **common,
                "value": "HKCategoryValueSleepAnalysisAsleepDeep",
                "startDate": "2026-07-02 03:00:00 +0800",
                "endDate": "2026-07-02 05:00:00 +0800",
            },
            {
                **common,
                "value": "HKCategoryValueSleepAnalysisAsleepREM",
                "startDate": "2026-07-02 05:00:00 +0800",
                "endDate": "2026-07-02 07:00:00 +0800",
            },
        ]
        dataset = self.write_export(records=records)

        daily_sleep, stages, daily_in_bed = dataset.sleep_summary()

        self.assertEqual(daily_sleep.iloc[0], 8)
        self.assertEqual(daily_in_bed.iloc[0], 8)
        self.assertEqual(stages.iloc[0]["Core Sleep"], 4)
        self.assertEqual(stages.iloc[0]["Deep Sleep"], 2)
        self.assertEqual(stages.iloc[0]["REM Sleep"], 2)

    def test_workout_distance_energy_and_heart_rate_are_normalized(self):
        records = [
            self.quantity(
                "HKQuantityTypeIdentifierHeartRate",
                150,
                "count/min",
                "2026-07-01 09:15:00 +0800",
                "2026-07-01 09:15:01 +0800",
            ),
            self.quantity(
                "HKQuantityTypeIdentifierHeartRate",
                170,
                "count/min",
                "2026-07-01 09:45:00 +0800",
                "2026-07-01 09:45:01 +0800",
            ),
        ]
        workouts = [
            (
                {
                    "workoutActivityType": "HKWorkoutActivityTypeCycling",
                    "duration": "60",
                    "durationUnit": "min",
                    "startDate": "2026-07-01 09:00:00 +0800",
                    "endDate": "2026-07-01 10:00:00 +0800",
                    "sourceName": "Apple Watch",
                },
                [
                    {
                        "type": "HKQuantityTypeIdentifierActiveEnergyBurned",
                        "sum": "418.4",
                        "unit": "kJ",
                    },
                    {
                        "type": "HKQuantityTypeIdentifierDistanceCycling",
                        "sum": "10",
                        "unit": "km",
                    },
                ],
            )
        ]
        dataset = self.write_export(records=records, workouts=workouts)

        workout = dataset.workouts(max_heart_rate=200).iloc[0]

        self.assertEqual(workout["distance_km"], 10)
        self.assertAlmostEqual(workout["calories"], 100, places=1)
        self.assertEqual(workout["avg_heart_rate"], 160)
        self.assertEqual(workout["max_heart_rate"], 170)
        self.assertEqual(workout["intensity_percent_max"], 80)

    def test_empty_processed_exports_still_have_readable_headers(self):
        dataset = self.write_export()
        output_dir = Path(self.temp_dir.name) / "output"

        dataset.write_metric_exports(output_dir)

        self.assertEqual(
            (output_dir / "steps_data.csv").read_text().strip(),
            "date,value",
        )
        self.assertEqual(
            (output_dir / "sleep_daily.csv").read_text().strip(),
            "date,value,in_bed_hours",
        )


if __name__ == "__main__":
    unittest.main()
