import pandas as pd
import pandas.testing as tm
import pandera as pa
import pytest
import json
import logging
from src import val_data_handle_error  # Assuming the function is imported

# Configure logging to capture output, similar to the example image
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# --- Schema Definition ---
# Define a standard schema to use in the tests
schema = pa.DataFrameSchema({
    "id_col": pa.Column(pa.Int, pa.Check.greater_than(0), unique=True),
    "value_col": pa.Column(pa.String, pa.Check.isin(["Good", "Bad"])),
    "score_col": pa.Column(pa.Float, pa.Check.less_than(10.0), nullable=True)
})

def test_val_data_handle_error():
    """
    Tests the val_data_handle_error function for handling valid data, 
    invalid data, and duplicates, and ensures correct error reporting.
    """
    
    LOGGER.info("Starting test_val_data_handle_error...")
  
    # 1. Test Case: Invalid Data and Duplicates    
    # Data with:
    # - Row 0: Valid
    # - Row 1: Invalid (id_col < 0)
    # - Row 2: Duplicate of Row 0
    # - Row 3: Invalid (value_col not in ["Good", "Bad"])
    data_mixed = {
        "id_col": [101, -2, 101, 103], 
        "value_col": ["Good", "Bad", "Good", "Unknown"],
        "score_col": [5.5, 9.9, 5.5, 1.1] 
    }
    df_in_mixed = pd.DataFrame(data_mixed)

    # Expected Result:
    # 1. Row 2 (duplicate) is dropped first.
    # 2. Rows 1 (-2) and 3 ("Unknown") are invalid and dropped during validation.
    # 3. Only Row 0 (original index 0) remains.
    expected_data_validated = {
        "id_col": [101],
        "value_col": ["Good"],
        "score_col": [5.5]
    }
    expected_df_validated = pd.DataFrame(expected_data_validated)
    
    validated_df, error_cases_mixed = val_data_handle_error(df_in_mixed, schema)
    
    # Assertions for Mixed Data Case
    
    # A. Check the validated DataFrame (should have removed duplicates and invalid rows)
    tm.assert_frame_equal(validated_df.reset_index(drop=True), expected_df_validated.reset_index(drop=True), check_dtype=False)
    
    # B. Check that error cases were captured and formatted as JSON
    assert error_cases_mixed is not None
    assert isinstance(error_cases_mixed, str)
    
    error_dict = json.loads(error_cases_mixed)
    
    # C. Check the JSON structure and failure cases (indices 1 and 3 should have failed validation)
    assert "schema_errors" in error_dict
    assert "failure_cases" in error_dict["schema_errors"]
    
    # Check that at least two rows failed validation (index 1 for id_col, index 3 for value_col)
    failure_cases = error_dict["schema_errors"]["failure_cases"]
    assert len(failure_cases) >= 2 
    
    # Check that the invalid indices (1 and 3) are present in the error report
    failed_indices = [case['index'] for case in failure_cases]
    assert 1 in failed_indices
    assert 3 in failed_indices
    
    LOGGER.info("Mixed Data/Error Test Passed.")

    # 2. Test Case: Fully Valid Data (Fast Pass)
    data_valid = {
        "id_col": [1, 2, 3],
        "value_col": ["Good", "Bad", "Good"],
        "score_col": [1.0, 2.0, None] # Testing nullable
    }
    df_in_valid = pd.DataFrame(data_valid)
    
    validated_df_valid, error_cases_valid = val_data_handle_error(df_in_valid, schema)
    
    # Assertions for Valid Data Case
    
    # A. The output dataframe should match the input dataframe (fast-forward)
    tm.assert_frame_equal(validated_df_valid, df_in_valid)
    
    # B. Error cases should be None
    assert error_cases_valid is None
    
    LOGGER.info("Fully Valid Data Test Passed.")
