"""Unit tests for ResultsDatabase fields."""

import unittest
from pathlib import Path

import pytest

from qaoa_parameter_setting.utils.database.results_database import ResultsDatabase

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestDatabaseFields(unittest.TestCase):
    """Integration tests for ResultsDatabase fields."""

    @classmethod
    def setUpClass(cls):
        """Set up test database with real data."""
        # Check if data directory exists
        data_path = Path("data/examples")
        if not data_path.exists():
            raise unittest.SkipTest(f"Test data not found at {data_path}")
        
        cls.db = ResultsDatabase()
        cls.db.add_data(str(data_path))
        cls.df = cls.db.to_dataframe()

    def test_dataframe_not_empty(self):
        """Test that DataFrame is not empty."""
        self.assertGreater(len(self.df), 0, "DataFrame should contain data")

    def test_dataframe_has_required_columns(self):
        """Test that DataFrame has all required columns."""
        required_columns = [
            "instance",
            "trainer_config",
            "method",
            "depth",
            "energy",
            "trainer",
            "evaluation",
            "evaluation_label",
            "method_label",
            "with_aer",
            "source_file",
            "train_duration",
            "metadata",
            "result_index",
            "run_datetime",
            "result_key_index",
        ]
        
        for col in required_columns:
            self.assertIn(col, self.df.columns, f"Column '{col}' should be present")

    def test_trainer_config_field(self):
        """Test trainer_config field."""
        self.assertIn("trainer_config", self.df.columns)
        self.assertGreater(self.df["trainer_config"].nunique(), 0)
        
        # Check that trainer_config values end with .json
        sample_configs = self.df["trainer_config"].dropna().unique()[:5]
        for config in sample_configs:
            self.assertTrue(config.endswith(".json"), 
                          f"trainer_config '{config}' should end with .json")

    def test_method_field(self):
        """Test method field."""
        self.assertIn("method", self.df.columns)
        self.assertGreater(self.df["method"].nunique(), 0)
        
        # Check that method values end with .json
        sample_methods = self.df["method"].dropna().unique()[:5]
        for method in sample_methods:
            self.assertTrue(method.endswith(".json"), 
                          f"method '{method}' should end with .json")

    def test_evaluation_label_field(self):
        """Test evaluation_label field."""
        self.assertIn("evaluation_label", self.df.columns)
        
        # Check that evaluation_label has valid values
        valid_labels = ["SV", "MPS (Quimb)", "MPS (Aer)", "PP"]
        unique_labels = self.df["evaluation_label"].unique()
        
        for label in unique_labels:
            self.assertIn(label, valid_labels, 
                        f"evaluation_label '{label}' should be one of {valid_labels}")

    def test_method_label_field(self):
        """Test method_label field."""
        self.assertIn("method_label", self.df.columns)
        self.assertGreater(self.df["method_label"].nunique(), 0)
        
        # Check that method_label values are non-empty strings
        sample_labels = self.df["method_label"].dropna().unique()[:10]
        for label in sample_labels:
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    def test_metadata_field(self):
        """Test metadata field."""
        self.assertIn("metadata", self.df.columns)
        
        # Check that metadata is a dict
        if len(self.df) > 0:
            first_metadata = self.df["metadata"].iloc[0]
            self.assertIsInstance(first_metadata, dict)

    def test_print_methods_by_evaluation(self):
        """Test that print_methods_by_evaluation runs without error."""
        import io
        import sys
        
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            self.db.print_methods_by_evaluation()
            output = captured_output.getvalue()
            
            # Check that output contains expected content
            self.assertGreater(len(output), 0, "Should produce output")
        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    unittest.main()