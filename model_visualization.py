import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'eval_plots')

FEATURE_LABELS = ['Age', 'Height', 'Gender', 'Smoking', 'Asthma History',
                  'Temperature', 'Humidity', 'PM2.5', 'PM10']

def ensure_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)
    return PLOT_DIR

def plot_feature_importance(model, path=None):
    ensure_dir()
    path = path or os.path.join(PLOT_DIR, 'feature_importance.png')
    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([FEATURE_LABELS[i] for i in order], importances[order],
            color='#3b82f6')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest — Feature Importance (PEFR prediction)')
    for i, v in enumerate(importances[order]):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_pred_vs_actual(y_test, y_pred, path=None):
    ensure_dir()
    path = path or os.path.join(PLOT_DIR, 'pred_vs_actual.png')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, s=18, color='#3b82f6')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect prediction (y=x)')
    ax.set_xlabel('Actual PEFR (L/min)')
    ax.set_ylabel('Predicted PEFR (L/min)')
    ax.set_title('Predicted vs Actual PEFR')
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_residuals(y_test, y_pred, path=None):
    ensure_dir()
    path = path or os.path.join(PLOT_DIR, 'residuals.png')
    residuals = np.array(y_test) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(residuals, bins=40, color='#8b5cf6', edgecolor='white')
    axes[0].axvline(0, color='red', ls='--', lw=1.5)
    axes[0].set_xlabel('Residual (Actual − Predicted) L/min')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Residuals')

    axes[1].scatter(y_pred, residuals, alpha=0.4, s=18, color='#8b5cf6')
    axes[1].axhline(0, color='red', ls='--', lw=1.5)
    axes[1].set_xlabel('Predicted PEFR (L/min)')
    axes[1].set_ylabel('Residual (L/min)')
    axes[1].set_title('Residuals vs Predicted (heteroscedasticity check)')

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_all(model, y_test, y_pred):
    paths = {
        'feature_importance': plot_feature_importance(model),
        'pred_vs_actual': plot_pred_vs_actual(y_test, y_pred),
        'residuals': plot_residuals(y_test, y_pred),
    }
    print('\n[6] Visualizations saved to eval_plots/:')
    for name, p in paths.items():
        print(f'    {name:20s} -> {os.path.basename(p)}')
    return paths
