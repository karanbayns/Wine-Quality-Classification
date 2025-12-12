import pandas as pd
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train_evaluate_models import train_evaluate_models

def test_train_evaluate_models():
    X = pd.DataFrame(
        {
            "fixed acidity": [7.0, 6.0, 8.0, 5.0],
            "volatile acidity": [0.7, 0.3, 0.5, 0.4],
            "citric acid": [0.0, 0.2, 0.3, 0.1],
            "residual sugar": [1.9, 6.0, 2.5, 3.0],
            "chlorides": [0.076, 0.045, 0.050, 0.060],
            "free sulfur dioxide": [11.0, 30.0, 20.0, 25.0],
            "total sulfur dioxide": [34.0, 100.0, 80.0, 60.0],
            "density": [0.9978, 0.994, 0.996, 0.995],
            "pH": [3.51, 3.30, 3.40, 3.45],
            "sulphates": [0.56, 0.65, 0.60, 0.55],
            "alcohol": [9.4, 11.0, 10.5, 9.8],
        }
    )
    y = pd.Series([0, 1, 1, 0], name="quality_binary")

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=123, max_iter=1000, class_weight="balanced"
        )
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train_scaled, X_test_scaled = X_scaled[:2], X_scaled[2:]
    y_train, y_test = y[:2], y[2:]

    results, trained_models = train_evaluate_models(
        models, X_train_scaled, y_train, X_test_scaled, y_test
    )

    assert isinstance(results, list)
    assert isinstance(trained_models, dict)
    assert "Logistic Regression" in trained_models
    assert set(trained_models.keys()) == set(models.keys())
    assert len(results) == len(models)
    row = results[0]
    for key in ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        assert 0.0 <= row[key] <= 1.0
