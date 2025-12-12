import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def plot_roc_curves(trained_models, X_test, y_test, path_save, filename = 'roc_curves.png', figsize = (10, 7)):
    if not trained_models:
        raise ValueError("trained_models dictionary is empty.")

    os.makedirs(path_save, exist_ok = True)

    # Plot ROC curves
    fig, ax = plt.subplots(figsize = figsize)

    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
    
        ax.plot(fpr, tpr, linewidth = 2.5, 
            label = f'{name} (AUC = {roc_auc:.3f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth = 2, label = 'Random Classifier (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate', fontsize = 12)
    ax.set_ylabel('True Positive Rate', fontsize = 12)
    ax.set_title('ROC Curves for Binary Wine Quality Classification', 
                 fontsize = 14, fontweight = 'bold')
    ax.legend(loc = 'lower right', fontsize = 11)
    ax.grid(alpha = 0.3)
    plt.tight_layout()

    output_path = os.path.join(path_save, filename)
    fig.savefig(output_path, dpi = 300, bbox_inches = 'tight')

    return fig, ax