import click
import numpy as np
import pandas as pd
import json
import pandera.pandas as pa
import logging
from pandera import extensions
from sklearn.model_selection import train_test_split
from scipy.stats import kstest, norm

@click.command()
@click.argument('path_read', type = str)
@click.argument('path_save', type = str)
@click.option('--delim', type = str)
def main(path_read, path_save, delim = ","):
    # Read the csv file from path_read, which can be a URL or filepath
    df = pd.read_csv(path_read, sep=delim)
    df['quality_binary'] = df['quality'] >= 7
    
    # Register custom checks
    @extensions.register_check_method(statistics = ['iqr_mult'])
    def outlier_check(df, iqr_mult = 3):
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        return (df >= lower) & (df <= upper)

    @extensions.register_check_method()
    def dist_check(df):
        z = (df - df.mean()) / df.std()
        stat, p = kstest(z, 'norm')
        return p > 0.05

    @extensions.register_check_method()
    def target_corr_check(df):
        corr = df.corr(numeric_only = True)['quality_binary'].drop('quality_binary')
        return corr.abs().max() < 0.95

    @extensions.register_check_method()
    def feature_corr_check(df):
        corr = df.drop(columns = 'quality_binary').corr(numeric_only = True)
        upper = np.triu(corr, k = 1)
        return np.abs(upper).max() < 0.95
        
    # Build schema
    # Used pa.Check.outlier_check() to check outliers in numerical features
    # nullable = False to check for null values
    schema = pa.DataFrameSchema(
        {
            'fixed acidity': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'volatile acidity': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'citric acid': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'residual sugar': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'chlorides': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'free sulfur dioxide': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'total sulfur dioxide': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'density': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'pH': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'sulphates': pa.Column(float, pa.Check.outlier_check(),nullable = False),
            'alcohol': pa.Column(float, pa.Check.outlier_check(),nullable = False),      
            'quality': pa.Column(int, checks = [pa.Check.between(0, 10), pa.Check.dist_check()],nullable = False),
            'quality_binary': pa.Column(bool, nullable = False)
        },
        checks = [ # Check duplicate rows
                pa.Check(lambda df: ~df.duplicated().any(), element_wise = False, error = 'Duplicate rows detected.')],
                drop_invalid_rows = False)

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
        col0   col1
        0     1
        1     2
        """
        error_cases = None
        try:
            schema.validate(df, lazy = True)
        except pa.errors.SchemaErrors as e:
            error_cases = e.failure_cases
    
            # Convert the error message to a JSON string
            error_message = json.dumps(e.message, indent = 2)
            logging.error('\n' + error_message)
    
        # Filter out invalid rows based on the error cases
        if error_cases is not None and not error_cases.empty:
            invalid_indices = error_cases['index'].dropna().unique()
            return (df.drop(index = invalid_indices)
                   .reset_index(drop = True)
                   .drop_duplicates()
                   .dropna(how = 'all'))
        else:
            return df
    
    # Validate the dataframe
    validated_df = val_data_handle_error(df, schema)
    
        # Separate features and target
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']

    X = validated_df[feature_columns]
    y = validated_df['quality_binary']

        # Split into train and test
    train_df, test_df = train_test_split(validated_df, test_size = 0.2, random_state = 2025, 
                                             stratify = validated_df['quality_binary'])

    schema_corr = pa.DataFrameSchema(
    {'fixed acidity': pa.Column(float),
    'volatile acidity': pa.Column(float),
    'citric acid': pa.Column(float),
    'residual sugar': pa.Column(float),
    'chlorides': pa.Column(float),
    'free sulfur dioxide': pa.Column(float),
    'total sulfur dioxide': pa.Column(float),
    'density': pa.Column(float),
    'pH': pa.Column(float),
    'sulphates': pa.Column(float),
    'alcohol': pa.Column(float),
    'quality': pa.Column(int),
    'quality_binary':   pa.Column(bool)},
     checks = [pa.Check.target_corr_check(),
              pa.Check.feature_corr_check()],
     drop_invalid_rows = False)
    
    train_df = val_data_handle_error(train_df, schema_corr)
    train_df.to_csv(path_save+"/train_data.csv", index = False)
    test_df.to_csv(path_save+"/test_data.csv", index = False)    

if __name__ == '__main__':
    main()