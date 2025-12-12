import os
import matplotlib.figure
import matplotlib.axes
import pytest
import sys
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.plot_roc import plot_roc_curves

X, y = make_classification(n_samples = 100, n_features = 5, n_classes = 2, random_state = 14)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 14)

model = LogisticRegression(max_iter = 1000, random_state = 14)
model.fit(X_train, y_train)

trained_models = {'Logistic Regression': model}

def test_file_creation(tmp_path):
    plot_roc_curves(trained_models, X_test, y_test, tmp_path)
    assert os.path.exists(tmp_path / 'roc_curves.png')

def test_return_types(tmp_path):
    fig, ax = plot_roc_curves(trained_models, X_test, y_test, tmp_path)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

def test_empty_models_dict(tmp_path):
    with pytest.raises(ValueError, match = 'trained_models dictionary is empty'):
        plot_roc_curves({}, X_test, y_test, tmp_path)

def test_multiple_lines(tmp_path):
    models = {
        'Model 1': model,
        'Model 2': LogisticRegression(max_iter = 1000, random_state = 5).fit(X_train, y_train)
    }

    fig, ax = plot_roc_curves(models, X_test, y_test, tmp_path)

    assert len(ax.lines) == 3 # 2 models + diagonal line
