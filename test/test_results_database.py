"""Comprehensive unit tests for ResultsDatabase class."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pytest

from qaoa_parameter_setting.utils.database.results_database import ResultsDatabase

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestResultsDatabase(unittest.TestCase):
    """Test suite for ResultsDatabase class."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = ResultsDatabase()

    def test_init_empty_database(self):
        """Test initialization of empty database."""
        db = ResultsDatabase()
        self.assertEqual(len(db.data), 0)
        self.assertEqual(len(db.source_files), 0)
        self.assertEqual(len(db.additional_methods), 0)

    def test_init_from_file(self):
        """Test initialization from saved file."""
        # Create test data
        test_data = {
            "data": {
                "test_instance.json": {
                    "FA_SV_opt.json": {
                        "5": [
                            {
                                "energy": 10.5,
                                "run_datetime": "2026-04-17T12:00:00",
                                "result_key_index": 0,
                            }
                        ]
                    }
                }
            },
            "source_files": ["test_file.json"],
            "additional_methods": ["FA_SV_no_opt.json"],
        }

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            # Load database from file
            db = ResultsDatabase(temp_file)

            # Verify data loaded correctly
            self.assertIn("test_instance.json", db.data)
            self.assertIn("FA_SV_opt.json", db.data["test_instance.json"])
            self.assertIn(5, db.data["test_instance.json"]["FA_SV_opt.json"])
            self.assertEqual(len(db.source_files), 1)
            self.assertIn("test_file.json", db.source_files)
            self.assertEqual(len(db.additional_methods), 1)
            self.assertIn("FA_SV_no_opt.json", db.additional_methods)

            # Verify datetime was parsed
            result = db.data["test_instance.json"]["FA_SV_opt.json"][5][0]
            self.assertIsInstance(result["run_datetime"], datetime)
        finally:
            # Clean up
            Path(temp_file).unlink()

    def test_convert_str_depth_to_int(self):
        """Test conversion of string depths to integers."""
        str_data = {
            "instance1.json": {
                "method1.json": {"5": [{"energy": 10.0}], "10": [{"energy": 20.0}]}
            }
        }

        result = self.db._ResultsDatabase__convert_str_depth_to_int(str_data)

        self.assertIn(5, result["instance1.json"]["method1.json"])
        self.assertIn(10, result["instance1.json"]["method1.json"])
        self.assertNotIn("5", result["instance1.json"]["method1.json"])

    def test_convert_str_depth_to_int_and_parse_datetime(self):
        """Test conversion of depths and datetime parsing."""
        str_data = {
            "instance1.json": {
                "method1.json": {
                    "5": [
                        {
                            "energy": 10.0,
                            "run_datetime": "2026-04-17T12:00:00",
                        }
                    ]
                }
            }
        }

        result = self.db._ResultsDatabase__convert_str_depth_to_int_and_parse_datetime(
            str_data
        )

        self.assertIn(5, result["instance1.json"]["method1.json"])
        result_entry = result["instance1.json"]["method1.json"][5][0]
        self.assertIsInstance(result_entry["run_datetime"], datetime)
        self.assertEqual(result_entry["run_datetime"].year, 2026)
        self.assertEqual(result_entry["run_datetime"].month, 4)
        self.assertEqual(result_entry["run_datetime"].day, 17)


class TestExtractDatetimeFromFilename(unittest.TestCase):
    """Test suite for extract_datetime_from_filename static method."""

    def test_valid_datetime_extraction(self):
        """Test extraction of valid datetime from filename."""
        filename = "20260417_123045_results.json"
        dt = ResultsDatabase.extract_datetime_from_filename(filename)

        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 17)
        self.assertEqual(dt.hour, 12)
        self.assertEqual(dt.minute, 30)
        self.assertEqual(dt.second, 45)

    def test_datetime_extraction_with_path(self):
        """Test extraction from full path."""
        filename = "/path/to/20260417_123045_results.json"
        dt = ResultsDatabase.extract_datetime_from_filename(filename)

        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 17)

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with self.assertRaises(ValueError) as context:
            ResultsDatabase.extract_datetime_from_filename("invalid_filename.json")

        self.assertIn("Could not extract datetime", str(context.exception))

    def test_invalid_date_raises_error(self):
        """Test that invalid date values raise ValueError."""
        with self.assertRaises(ValueError):
            ResultsDatabase.extract_datetime_from_filename(
                "20261332_123045_results.json"
            )


class TestGetConfigInfo(unittest.TestCase):
    """Test suite for _get_config_info method."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = ResultsDatabase()

    def test_config_info_structure(self):
        """Test that config info has correct structure."""
        config = "FA_SV_opt.json"
        info = self.db._get_config_info(config)

        self.assertIn("evaluation", info)
        self.assertIn("method", info)
        self.assertIn("evaluation_label", info)
        self.assertIn("with_aer", info)
        self.assertIn("method_label", info)

    def test_config_info_values(self):
        """Test that config info has correct values."""
        config = "FA_SV_opt.json"
        info = self.db._get_config_info(config)

        self.assertEqual(info["evaluation"], "SV")
        self.assertEqual(info["method"], "FA_opt.json")
        self.assertEqual(info["evaluation_label"], "SV")
        self.assertFalse(info["with_aer"])


class TestListMethods(unittest.TestCase):
    """Test suite for list_methods method."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.db = ResultsDatabase()
        # Add some test data
        self.db._data["instance1.json"]["FA_SV_opt.json"][5] = [{"energy": 10.0}]
        self.db._data["instance1.json"]["TQA_MPS_opt.json"][5] = [{"energy": 20.0}]
        self.db._additional_methods.add("FA_SV_no_opt.json")

    def test_list_methods_returns_all_methods(self):
        """Test that list_methods returns all unique methods."""
        methods = self.db.list_methods()

        self.assertIn("FA_SV_opt.json", methods)
        self.assertIn("TQA_MPS_opt.json", methods)
        self.assertIn("FA_SV_no_opt.json", methods)
        self.assertEqual(len(methods), 3)


class TestSaveAndLoad(unittest.TestCase):
    """Test suite for save and load functionality."""

    def test_save_and_load_roundtrip(self):
        """Test that data can be saved and loaded correctly."""
        # Create database with test data
        db1 = ResultsDatabase()
        db1._data["instance1.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.5,
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,
            }
        ]
        db1._source_files.add("test_file.json")
        db1._additional_methods.add("FA_SV_no_opt.json")

        # Create temporary file path (but don't create the file yet)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
            temp_file = f.name
        # File is now deleted, so save() won't complain

        try:
            db1.save(temp_file)

            # Load from file
            db2 = ResultsDatabase(temp_file)

            # Verify data matches
            self.assertEqual(
                db1.data["instance1.json"]["FA_SV_opt.json"][5][0]["energy"],
                db2.data["instance1.json"]["FA_SV_opt.json"][5][0]["energy"],
            )
            self.assertEqual(db1.source_files, db2.source_files)
            self.assertEqual(db1.additional_methods, db2.additional_methods)

            # Verify datetime was preserved
            dt1 = db1.data["instance1.json"]["FA_SV_opt.json"][5][0]["run_datetime"]
            dt2 = db2.data["instance1.json"]["FA_SV_opt.json"][5][0]["run_datetime"]
            self.assertEqual(dt1, dt2)
        finally:
            # Clean up
            Path(temp_file).unlink()


class TestFailedConfigsJsonMethods(unittest.TestCase):
    """Test suite for load_failed_configs_from_json and save_failed_configs_to_json static methods."""

    def test_load_converts_string_depths_to_integers(self):
        """Test that load_failed_configs_from_json converts string depth keys to integers."""
        # Create test JSON file
        json_data = {
            "instance1.json": {
                "method1.json": {
                    "1": "reason1",
                    "2": "reason2",
                    "10": "reason10",
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_file = f.name

        try:
            result = ResultsDatabase.load_failed_configs_from_json(temp_file)

            # Check that depths are now integers
            self.assertIn(1, result["instance1.json"]["method1.json"])
            self.assertIn(2, result["instance1.json"]["method1.json"])
            self.assertIn(10, result["instance1.json"]["method1.json"])

            # Check that string keys are gone
            self.assertNotIn("1", result["instance1.json"]["method1.json"])
            self.assertNotIn("2", result["instance1.json"]["method1.json"])
        finally:
            Path(temp_file).unlink()

    def test_load_preserves_reasons(self):
        """Test that failure reasons are preserved when loading."""
        json_data = {
            "instance1.json": {
                "method1.json": {
                    "5": "No FA for P=5 and avg. degree. 8.",
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_data, f)
            temp_file = f.name

        try:
            result = ResultsDatabase.load_failed_configs_from_json(temp_file)

            self.assertEqual(
                result["instance1.json"]["method1.json"][5],
                "No FA for P=5 and avg. degree. 8.",
            )
        finally:
            Path(temp_file).unlink()

    def test_save_and_load_roundtrip(self):
        """Test that data can be saved and loaded correctly."""
        failed_configs = {
            "instance1.json": {
                "method1.json": {1: "reason1", 2: "reason2"},
                "method2.json": {3: "reason3"},
            },
            "instance2.json": {
                "method3.json": {4: "reason4"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Save
            ResultsDatabase.save_failed_configs_to_json(failed_configs, temp_file)

            # Load
            loaded = ResultsDatabase.load_failed_configs_from_json(temp_file)

            # Verify data matches
            self.assertEqual(len(loaded), 2)
            self.assertEqual(len(loaded["instance1.json"]), 2)
            self.assertEqual(len(loaded["instance2.json"]), 1)
            self.assertEqual(loaded["instance1.json"]["method1.json"][1], "reason1")
            self.assertEqual(loaded["instance1.json"]["method1.json"][2], "reason2")
            self.assertEqual(loaded["instance1.json"]["method2.json"][3], "reason3")
            self.assertEqual(loaded["instance2.json"]["method3.json"][4], "reason4")
        finally:
            Path(temp_file).unlink()

    def test_save_creates_valid_json(self):
        """Test that saved file is valid JSON with string keys."""
        failed_configs = {
            "instance1.json": {"method1.json": {1: "reason1", 2: "reason2"}}
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            ResultsDatabase.save_failed_configs_to_json(failed_configs, temp_file)

            # Load with standard json.load to verify format
            with open(temp_file) as f:
                json_data = json.load(f)

            # Verify keys are strings in JSON
            self.assertIn("1", json_data["instance1.json"]["method1.json"])
            self.assertIn("2", json_data["instance1.json"]["method1.json"])
            self.assertNotIn(1, json_data["instance1.json"]["method1.json"])
        finally:
            Path(temp_file).unlink()

    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        failed_configs = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            ResultsDatabase.save_failed_configs_to_json(failed_configs, temp_file)
            loaded = ResultsDatabase.load_failed_configs_from_json(temp_file)
            self.assertEqual(loaded, {})
        finally:
            Path(temp_file).unlink()


if __name__ == "__main__":
    unittest.main()


class TestConfigPathToConfig(unittest.TestCase):
    """Test suite for config_path_to_config method."""

    def setUp(self):
        """Set up test fixtures."""
        self.db = ResultsDatabase()

    def test_extracts_filename_from_path(self):
        """Test extraction of filename from full path."""
        config_path = "methods/FA_SV_opt.json"
        result = self.db.config_path_to_config(config_path)
        self.assertEqual(result, "FA_SV_opt.json")

    def test_handles_filename_only(self):
        """Test handling of filename without path."""
        config_path = "FA_SV_opt.json"
        result = self.db.config_path_to_config(config_path)
        self.assertEqual(result, "FA_SV_opt.json")


class TestGetIterKeys(unittest.TestCase):
    """Test suite for get_iter_keys static method."""

    def test_extracts_numeric_keys(self):
        """Test extraction of numeric iteration keys."""
        keys = ["0", "1", "2", "args", "metadata", "10"]
        result = ResultsDatabase.get_iter_keys(keys)
        self.assertEqual(set(result), {"0", "1", "2", "10"})

    def test_filters_non_numeric_keys(self):
        """Test that non-numeric keys are filtered out."""
        keys = ["args", "config", "metadata"]
        result = ResultsDatabase.get_iter_keys(keys)
        self.assertEqual(result, [])

    def test_handles_empty_list(self):
        """Test handling of empty key list."""
        keys = []
        result = ResultsDatabase.get_iter_keys(keys)
        self.assertEqual(result, [])


class TestPrintMethodsByEvaluation(unittest.TestCase):
    """Test suite for print_methods_by_evaluation method."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.db = ResultsDatabase()
        self.db._data["instance1.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {},
                "run_datetime": datetime(2026, 4, 17),
                "result_key_index": 0,
            }
        ]
        self.db._data["instance1.json"]["TQA_MPS_opt.json"][5] = [
            {
                "energy": 20.0,
                "trainer": "TQA",
                "evaluation": "MPS",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {},
                "run_datetime": datetime(2026, 4, 17),
                "result_key_index": 0,
            }
        ]

    def test_prints_methods_table(self):
        """Test that print_methods_by_evaluation produces output."""
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            self.db.print_methods_by_evaluation()
            output = captured_output.getvalue()
            self.assertIn("MPS (Quimb)", output)
            self.assertIn("SV", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_handles_empty_database(self):
        """Test handling of empty database."""
        import io
        import sys

        empty_db = ResultsDatabase()
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            empty_db.print_methods_by_evaluation()
            output = captured_output.getvalue()
            self.assertIn("No methods found", output)
        finally:
            sys.stdout = sys.__stdout__


class TestFilterByInstanceEvaluation(unittest.TestCase):
    """Test suite for filter_by_instance_evaluation method."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.db = ResultsDatabase()
        # Add SV method data
        self.db._data["instance1.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {},
                "run_datetime": datetime(2026, 4, 17),
                "result_key_index": 0,
            }
        ]
        # Add MPS method data (non-Aer)
        self.db._data["instance2.json"]["TQA_MPS_opt.json"][5] = [
            {
                "energy": 20.0,
                "trainer": "TQA",
                "evaluation": "MPS",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {},
                "run_datetime": datetime(2026, 4, 17),
                "result_key_index": 0,
            }
        ]
        # Add MPS method data (Aer)
        self.db._data["instance3.json"]["FA_MPSAer_opt.json"][5] = [
            {
                "energy": 30.0,
                "trainer": "FA",
                "evaluation": "MPS",
                "with_aer": True,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {},
                "run_datetime": datetime(2026, 4, 17),
                "result_key_index": 0,
            }
        ]

    def test_filter_by_simple_set(self):
        """Test filtering with simple set of instances."""
        instance_filter = {"SV": {"instance1.json"}}
        filtered = self.db.filter_by(instance_filter=instance_filter)

        self.assertIn("instance1.json", filtered.data)
        self.assertNotIn("instance2.json", filtered.data)
        self.assertNotIn("instance3.json", filtered.data)

    def test_filter_by_aer_dict(self):
        """Test filtering with Aer-based dict."""
        instance_filter = {"MPS": {False: {"instance2.json"}}}
        filtered = self.db.filter_by(instance_filter=instance_filter)

        self.assertNotIn("instance1.json", filtered.data)
        self.assertIn("instance2.json", filtered.data)
        self.assertNotIn("instance3.json", filtered.data)

    def test_filter_preserves_source_files(self):
        """Test that filtering preserves source files."""
        self.db._source_files.add("test_file.json")
        instance_filter = {"SV": {"instance1.json"}}
        filtered = self.db.filter_by(instance_filter=instance_filter)

        self.assertEqual(filtered.source_files, self.db.source_files)


class TestToDataFrame(unittest.TestCase):
    """Test suite for to_dataframe method."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.db = ResultsDatabase()
        self.db._data["000_10nodes_random3regular.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.5,
                "trainer": "FA",
                "trainer_config": "FA_SV_opt.json",
                "method": "FA_opt.json",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.5,
                "metadata": {"iteration": "0", "evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,
            }
        ]

    def test_creates_dataframe_with_correct_columns(self):
        """Test that DataFrame has all expected columns."""
        df = self.db.to_dataframe()

        expected_columns = [
            "instance",
            "trainer_config",
            "method",
            "depth",
            "energy",
            "trainer",
            "evaluation",
            "evaluation_label",
            "evaluator",
            "method_label",
            "with_aer",
            "source_file",
            "train_duration",
            "metadata",
            "result_index",
            "run_datetime",
            "result_key_index",
            "mps_bond_dimension",
            "mps_threshold",
            "pp_max_weight",
            "pp_min_abs_coeff",
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns)

    def test_dataframe_has_correct_values(self):
        """Test that DataFrame contains correct values."""
        df = self.db.to_dataframe()


class TestOnlyBestParameters(unittest.TestCase):
    """Test suite for only_best_parameters method."""

    def setUp(self):
        """Set up test fixtures with sample data."""
        self.db = ResultsDatabase()

        # Add multiple results for same instance/depth with different energies
        self.db._data["000_10nodes_random3regular.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.5,
                "trainer": "FA",
                "trainer_config": "FA_SV_opt.json",
                "method": "FA_opt.json",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "test1.json",
                "train_duration": 1.5,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,
            }
        ]

        self.db._data["000_10nodes_random3regular.json"]["TQA_SV_opt.json"][5] = [
            {
                "energy": 12.0,  # Higher energy - should be selected for "instance" mode
                "trainer": "TQA",
                "trainer_config": "TQA_SV_opt.json",
                "method": "TQA_opt.json",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "TQA*",
                "with_aer": False,
                "source_file": "test2.json",
                "train_duration": 2.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 13, 0, 0),
                "result_key_index": 0,
            }
        ]

        # Add result for different evaluation
        self.db._data["000_10nodes_random3regular.json"]["FA_MPS_opt.json"][5] = [
            {
                "energy": 15.0,
                "trainer": "FA",
                "trainer_config": "FA_MPS_opt.json",
                "method": "FA_opt.json",
                "evaluation": "MPS",
                "evaluation_label": "MPS (Quimb)",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "test3.json",
                "train_duration": 1.5,
                "metadata": {
                    "evaluator": "MPSEvaluator",
                    "mps_bond_dimension": 20,
                    "mps_threshold": 1e-6,
                },
                "run_datetime": datetime(2026, 4, 17, 14, 0, 0),
                "result_key_index": 0,
            }
        ]

    def test_by_instance_selects_best_per_evaluation(self):
        """Test that by='instance' selects best result per (instance, evaluation, depth)."""
        filtered = self.db.only_best_parameters(by="instance")

        # Should have 2 results: best SV (TQA with 12.0) and best MPS (FA with 15.0)
        df = filtered.to_dataframe()
        self.assertEqual(len(df), 2)

        # Check SV result is TQA (higher energy)
        sv_results = df[df["evaluation_label"] == "SV"]
        self.assertEqual(len(sv_results), 1)
        self.assertEqual(sv_results.iloc[0]["trainer"], "TQA")
        self.assertEqual(sv_results.iloc[0]["energy"], 12.0)

        # Check MPS result
        mps_results = df[df["evaluation_label"] == "MPS (Quimb)"]
        self.assertEqual(len(mps_results), 1)
        self.assertEqual(mps_results.iloc[0]["energy"], 15.0)

    def test_by_config_keeps_best_per_method(self):
        """Test that by='config' keeps best result per (instance, evaluation, method, depth)."""
        filtered = self.db.only_best_parameters(by="config")

        # Should have 3 results: one for each method/evaluation combination
        df = filtered.to_dataframe()
        self.assertEqual(len(df), 3)

        # Each method should be present
        methods = set(df["method_label"])
        self.assertIn("Fixed Angle*", methods)
        self.assertIn("TQA*", methods)

    def test_tie_breaking_by_datetime(self):
        """Test tie-breaking when multiple results have same max energy."""
        # Add two results with same energy but different datetimes
        self.db._data["000_20nodes_random3regular.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "old.json",
                "train_duration": 1.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 1, 12, 0, 0),  # Older
                "result_key_index": 0,
            },
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "new.json",
                "train_duration": 1.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(
                    2026, 4, 17, 12, 0, 0
                ),  # Newer - should be selected
                "result_key_index": 0,
            },
        ]

        filtered = self.db.only_best_parameters(by="instance")
        df = filtered.to_dataframe()

        # Check that newer result was selected
        instance2_results = df[df["instance"] == "000_20nodes_random3regular.json"]
        self.assertEqual(len(instance2_results), 1)
        self.assertEqual(instance2_results.iloc[0]["source_file"], "new.json")

    def test_tie_breaking_by_result_key_index(self):
        """Test tie-breaking by result_key_index when datetime is same."""
        # Add two results with same energy and datetime but different result_key_index
        self.db._data["000_30nodes_random3regular.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "iter2.json",
                "train_duration": 1.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 2,  # Higher index
            },
            {
                "energy": 10.0,
                "trainer": "FA",
                "evaluation": "SV",
                "evaluation_label": "SV",
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "iter0.json",
                "train_duration": 1.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,  # Lower index - should be selected
            },
        ]

        filtered = self.db.only_best_parameters(by="instance")
        df = filtered.to_dataframe()

        # Check that result with lower index was selected
        instance3_results = df[df["instance"] == "000_30nodes_random3regular.json"]
        self.assertEqual(len(instance3_results), 1)
        self.assertEqual(instance3_results.iloc[0]["source_file"], "iter0.json")

    def test_invalid_by_parameter_raises_error(self):
        """Test that invalid 'by' parameter raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.db.only_best_parameters(by="invalid")

        self.assertIn("Invalid value for 'by'", str(context.exception))

    def test_preserves_source_files_and_additional_methods(self):
        """Test that filtering preserves source files and additional methods."""
        self.db._source_files.add("test_file.json")
        self.db._additional_methods.add("FA_SV_no_opt.json")

        filtered = self.db.only_best_parameters(by="instance")

        self.assertEqual(filtered.source_files, self.db.source_files)
        self.assertEqual(filtered.additional_methods, self.db.additional_methods)

    def test_handles_missing_evaluation_label(self):
        """Test handling of results without evaluation_label (fallback computation)."""
        # Add result without evaluation_label
        self.db._data["000_40nodes_random3regular.json"]["FA_PP_opt.json"][5] = [
            {
                "energy": 20.0,
                "trainer": "FA",
                "evaluation": "PP",
                # No evaluation_label - should be computed
                "method_label": "Fixed Angle*",
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {
                    "evaluator": "PPEvaluator",
                    "pp_max_weight": 4,
                    "pp_min_abs_coeff": 0.01,
                },
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,
            }
        ]

        filtered = self.db.only_best_parameters(by="instance")
        df = filtered.to_dataframe()

        # Should successfully process the result
        instance4_results = df[df["instance"] == "000_40nodes_random3regular.json"]
        self.assertEqual(len(instance4_results), 1)

    def test_handles_missing_method_label(self):
        """Test handling of results without method_label (fallback computation)."""
        # Create a fresh database for this test
        db = ResultsDatabase()
        db._data["000_50nodes_random3regular.json"]["FA_SV_opt.json"][5] = [
            {
                "energy": 20.0,
                "trainer": "FA",
                "evaluation": "SV",
                "evaluation_label": "SV",
                # No method_label - should be computed
                "with_aer": False,
                "source_file": "test.json",
                "train_duration": 1.0,
                "metadata": {"evaluator": "StatevectorEvaluator"},
                "run_datetime": datetime(2026, 4, 17, 12, 0, 0),
                "result_key_index": 0,
            }
        ]

        filtered = db.only_best_parameters(by="config")
        df = filtered.to_dataframe()

        # Should successfully process the result
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["instance"], "000_50nodes_random3regular.json")
        self.assertEqual(df.iloc[0]["depth"], 5)
        self.assertEqual(df.iloc[0]["trainer"], "FA")

    def test_empty_database_returns_empty_dataframe(self):
        """Test that empty database returns empty DataFrame."""
        empty_db = ResultsDatabase()
        df = empty_db.to_dataframe()

        self.assertEqual(len(df), 0)
