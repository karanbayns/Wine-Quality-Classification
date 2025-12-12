import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)

def train_evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test):
    """
    The function aim to train classification models and compute evaluation metrics.

    Parameters
    ----------
    models : dict
        A dictionary where the keys are model names and the values are untrained sklearn classifier objects.

    X_train_scaled : pandas.DataFrame
        Standardized training features, with one row per sample and one column per feature.

    y_train : pandas.Series
        Training labels as a one-dimensional vector of binary values.

    X_test_scaled : pandas.DataFrame
        Standardized test features, with one row per sample and one column per feature.

    y_test : pandas.Series
        Test labels as a one-dimensional vector of binary values.

    Returns
    -------
    results : list[dict]
        A list of dictionaries containing the evaluation metrics for each model (accuracy, precision, recall, F1 score, ROC AUC, etc.).

    trained_models : dict[str, sklearn.base.BaseEstimator]
    A dictionary of fitted models with the same keys as `models`, where each value is the corresponding trained classifier.
    """

    trained_models = {}
    results = []

    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_proba)

        # Store results
        results.append({
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

    return results, trained_models
