"""Unit tests for only_best_parameters() method."""

import unittest
from pathlib import Path

import pytest

from qaoa_parameter_setting.utils.database.results_database import ResultsDatabase

# Ignore warnings that the min-/max-cut data is missing for all tests
pytestmark = pytest.mark.filterwarnings(
    "ignore:Missing min-max cut data for instance.*:UserWarning"
)


class TestBestParametersIntegration(unittest.TestCase):
    """Integration tests for only_best_parameters() method."""

    @classmethod
    def setUpClass(cls):
        """Set up test database with real data."""
        # Check if data directory exists
        data_path = Path("data/examples")
        if not data_path.exists():
            raise unittest.SkipTest(f"Test data not found at {data_path}")
        
        cls.db = ResultsDatabase()
        cls.db.add_data(str(data_path))
        cls.original_df = cls.db.to_dataframe()

    def test_database_loaded_successfully(self):
        """Test that database loaded data successfully."""
        self.assertGreater(len(self.original_df), 0, "Database should contain results")
        self.assertGreater(self.original_df['instance'].nunique(), 0, "Database should contain instances")
        self.assertGreater(len(self.db.list_methods()), 0, "Database should contain methods")

    def test_only_best_parameters_by_instance(self):
        """Test only_best_parameters with by='instance'."""
        best_by_instance = self.db.only_best_parameters(by="instance")
        df_instance = best_by_instance.to_dataframe()
        
        # Should have fewer or equal results than original
        self.assertLessEqual(len(df_instance), len(self.original_df))
        
        # Should have same number of unique instances
        self.assertEqual(
            df_instance['instance'].nunique(),
            self.original_df['instance'].nunique()
        )
        
        # Check that for each (instance, evaluation, depth), only one result exists
        if len(df_instance) > 0:
            sample_instance = df_instance['instance'].iloc[0]
            sample_depth = df_instance['depth'].iloc[0]
            sample_eval = df_instance['evaluation_label'].iloc[0]
            
            sample_results = df_instance[
                (df_instance['instance'] == sample_instance) & 
                (df_instance['depth'] == sample_depth) &
                (df_instance['evaluation_label'] == sample_eval)
            ]
            
            # Should have exactly one result per (instance, evaluation, depth)
            self.assertEqual(len(sample_results), 1)

    def test_only_best_parameters_by_config(self):
        """Test only_best_parameters with by='config'."""
        best_by_config = self.db.only_best_parameters(by="config")
        df_config = best_by_config.to_dataframe()
        
        # Should have fewer or equal results than original
        self.assertLessEqual(len(df_config), len(self.original_df))
        
        # Should have same number of unique instances
        self.assertEqual(
            df_config['instance'].nunique(),
            self.original_df['instance'].nunique()
        )
        
        # Check that for each (instance, evaluation, method, depth), only one result exists
        if len(df_config) > 0:
            sample_instance = df_config['instance'].iloc[0]
            sample_depth = df_config['depth'].iloc[0]
            sample_method = df_config['method_label'].iloc[0]
            sample_eval = df_config['evaluation_label'].iloc[0]
            
            sample_results = df_config[
                (df_config['instance'] == sample_instance) & 
                (df_config['depth'] == sample_depth) &
                (df_config['method_label'] == sample_method) &
                (df_config['evaluation_label'] == sample_eval)
            ]
            
            # Should have exactly one result per (instance, evaluation, method, depth)
            self.assertEqual(len(sample_results), 1)

    def test_by_instance_has_fewer_results_than_by_config(self):
        """Test that by='instance' typically has fewer results than by='config'."""
        best_by_instance = self.db.only_best_parameters(by="instance")
        best_by_config = self.db.only_best_parameters(by="config")
        
        df_instance = best_by_instance.to_dataframe()
        df_config = best_by_config.to_dataframe()
        
        # by='instance' should have fewer or equal results than by='config'
        # because it selects best across all methods
        self.assertLessEqual(len(df_instance), len(df_config))


if __name__ == "__main__":
    unittest.main()