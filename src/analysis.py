import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score)
from src.train_evaluate_models import train_evaluate_models
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
            random_state=123, 
            max_iter=1000, 
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=123, 
            max_depth=10, 
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=123, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced'
        )
    }

    # Train models and store results
    results, trained_models = train_evaluate_models( models, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save the results as a dataframe
    pd.DataFrame(results).to_csv(path_save+"/model_performance_metrics.csv",index = False)

    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    
        ax.plot(fpr, tpr, linewidth=2.5, 
            label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Binary Wine Quality Classification', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(path_save + "/roc_curves.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()