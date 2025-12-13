import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.train_evaluate_models import train_evaluate_models
from src.plot_roc import plot_roc_curves
@click.command()
@click.argument('path_train', type = str)
@click.argument('path_test', type = str)
@click.argument('path_save', type = str)

def main(path_train, path_test, path_save):
    # Real the train and test datasets
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']  
    
    # Split into X and y
    X_train = train_df[feature_columns]
    y_train = train_df['quality_binary']

    X_test = test_df[feature_columns]
    y_test = test_df['quality_binary']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models with class balancing
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=2025, 
            max_iter=1000, 
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=2025, 
            max_depth=10, 
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=2025, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced'
        )
    }

    # Train models and store results
    results, trained_models = train_evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save the results as a dataframe
    pd.DataFrame(results).to_csv(path_save+"/model_performance_metrics.csv",index = False)

    # Plot ROC curves
    plot_roc_curves(trained_models, X_test_scaled, y_test, path_save)

if __name__ == '__main__':
    main()
