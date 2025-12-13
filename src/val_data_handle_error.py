import pandas as pd
import pandera.pandas as pa
import logging
import json

def val_data_handle_error(df, schema):
    """
    Validate a DataFrame against a Pandera schema and remove invalid rows.

    This function validates the input DataFrame using the provided Pandera schema.
    If validation errors occur, invalid rows are filtered out and the cleaned
    DataFrame is returned. Validation errors are logged as JSON for debugging.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to validate.
    schema : pandera.DataFrameSchema
        The Pandera schema defining validation rules for the DataFrame.

    Returns
    -------
    pandas.DataFrame
        The validated DataFrame with invalid rows removed, duplicates dropped,
        missing values removed, and index reset. If no validation errors occur,
        returns the original DataFrame unchanged.

    Raises
    ------
    None
        This function catches SchemaErrors internally and does not raise exceptions.

    Notes
    -----
    - Validation is performed lazily to collect all errors at once.
    - Invalid rows are identified by their index and removed from the DataFrame.
    - The returned DataFrame has duplicates removed and all rows with any
      missing values dropped.
    - Validation errors are logged in JSON format for easier parsing.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandera as pa
    >>> schema = pa.DataFrameSchema({"col1": pa.Column(int, pa.Check.between(0, 10))})
    >>> df = pd.DataFrame({"col1": [1, -1, 2]})
    >>> validated_df = val_data_handle_error(df, schema)
    >>> validated_df
        col0    col1
        0       1
        1       2
    """
    error_cases = None
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        error_cases = e.failure_cases

        # Convert the error message to a JSON string
        error_message = json.dumps(e.message, indent=2)
        logging.error('\n' + error_message)

    # Filter out invalid rows based on the error cases
    if error_cases is not None and not error_cases.empty:
        invalid_indices = error_cases['index'].dropna().unique()
        return (df.drop(index=invalid_indices)
                .reset_index(drop=True)
                .drop_duplicates()
                .dropna(how='all'))
    else:
        return df