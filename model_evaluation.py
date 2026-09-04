import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score, mean_absolute_percentage_error)
from sklearn.ensemble import RandomForestRegressor
from model_training import load_data, FEATURE_NAMES, TARGET, HYPERPARAMS

RANDOM_STATE = 42

PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, None],
    'min_samples_leaf': [2, 5],
    'min_samples_split': [2, 5],
}

def _metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

def load_xy():
    data = load_data()
    X = data[FEATURE_NAMES].values
    y = data[TARGET].values
    return data, X, y

def split_three_way(X, y):
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test

def tune_hyperparameters(X_train, y_train, fast=False):
    grid = PARAM_GRID
    if fast:
        grid = {'n_estimators': [100, 200], 'max_depth': [15, None]}

    combos = int(np.prod([len(v) for v in grid.values()]))
    print(f"  Tuning RandomForestRegressor over {combos} combos...")
    search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        param_grid=grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_ * -1

def evaluate_cv(model, n_splits=5):
    _, X, y = load_xy()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    rmse = -cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {
        'mae_mean': mae.mean(), 'mae_std': mae.std(),
        'rmse_mean': rmse.mean(), 'rmse_std': rmse.std(),
        'r2_mean': r2.mean(), 'r2_std': r2.std(),
    }

def compare_models():
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    _, X, y = load_xy()
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=RANDOM_STATE),
        'RandomForestRegressor (default)': RandomForestRegressor(**HYPERPARAMS),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    }
    results = {}
    for name, model in candidates.items():
        mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
        results[name] = {'mae_mean': mae.mean(), 'mae_std': mae.std(),
                         'r2_mean': r2.mean(), 'r2_std': r2.std()}
    return results

def subgroup_errors(model, X_test, y_test, data, test_idx):
    y_pred = model.predict(X_test)
    test_df = data.iloc[test_idx].copy()
    test_df['y_true'] = y_test
    test_df['y_pred'] = y_pred

    groups = {}
    groups['Gender'] = {'Male': test_df['Gender'] == 1, 'Female': test_df['Gender'] == 0}
    groups['Smoking'] = {'Smoker': test_df['Smoking'] == 1, 'Non-Smoker': test_df['Smoking'] == 0}
    groups['Asthma History'] = {'Asthmatic': test_df['AsthmaHistory'] == 1, 'Non-Asthmatic': test_df['AsthmaHistory'] == 0}

    age_bins = pd.cut(test_df['Age'], bins=[0, 30, 50, 100], labels=['<30', '30-50', '>50'])
    groups['Age Group'] = {label: (age_bins == label) for label in ['<30', '30-50', '>50']}

    pefr_q = pd.qcut(test_df['y_true'], q=3, labels=['Low PEFR', 'Mid PEFR', 'High PEFR'])
    groups['PEFR Level'] = {label: (pefr_q == label) for label in ['Low PEFR', 'Mid PEFR', 'High PEFR']}

    rows = []
    for group_name, subgroups in groups.items():
        for label, mask in subgroups.items():
            if mask.sum() == 0:
                continue
            mae = mean_absolute_error(test_df.loc[mask, 'y_true'], test_df.loc[mask, 'y_pred'])
            rows.append({'Group': group_name, 'Subgroup': label,
                         'n': int(mask.sum()), 'MAE': round(mae, 1)})
    return pd.DataFrame(rows)

def run_full_evaluation(verbose=True, fast=False):
    data, X, y = load_xy()
    idx = np.arange(len(y))
    X_train, X_rest, y_train, y_rest, idx_train, idx_rest = train_test_split(
        X, y, idx, test_size=0.30, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_rest, y_rest, idx_rest, test_size=0.50, random_state=RANDOM_STATE)

    print('=' * 60)
    print('MODEL EVALUATION — 3-WAY SPLIT + HYPERPARAMETER TUNING')
    print('=' * 60)
    print(f'  Train: {len(X_train)}  Validation: {len(X_val)}  Test: {len(X_test)}')

    print('\n[1] Hyperparameter tuning (GridSearchCV, 5-fold, on train only):')
    best_model, best_params, best_cv_mae = tune_hyperparameters(X_train, y_train, fast=fast)
    for k, v in best_params.items():
        print(f'    {k:18s} = {v}')
    print(f'    Best CV MAE (train folds): {best_cv_mae:.2f} L/min')

    print('\n[2] Validation sanity check (no overfit?):')
    val_metrics = _metrics(y_val, best_model.predict(X_val))
    print(f'    Validation MAE: {val_metrics["mae"]:.2f}  (CV train MAE: {best_cv_mae:.2f})')
    drift = abs(val_metrics['mae'] - best_cv_mae)
    print(f'    Gap: {drift:.2f} L/min  {"OK - no overfitting" if drift < 8 else "WARNING - possible overfit"}')

    print('\n[3] Final test set metrics (untouched):')
    y_pred = best_model.predict(X_test)
    test_metrics = _metrics(y_test, y_pred)
    for k, v in test_metrics.items():
        unit = '%' if k == 'mape' else ''
        print(f'    {k.upper():18s}: {v:.2f}{unit}')

    print('\n[4] 5-fold CV on full data (best params):')
    cv_metrics = evaluate_cv(best_model)
    print(f"    MAE : {cv_metrics['mae_mean']:.2f} ± {cv_metrics['mae_std']:.2f} L/min")
    print(f"    RMSE: {cv_metrics['rmse_mean']:.2f} ± {cv_metrics['rmse_std']:.2f} L/min")
    print(f"    R²  : {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")

    print('\n[5] Model comparison (5-fold CV, default params):')
    comparisons = compare_models()
    for name, m in comparisons.items():
        print(f"    {name:32s}: MAE {m['mae_mean']:.2f} ± {m['mae_std']:.2f}, R² {m['r2_mean']:.4f}")

    print('\n[6] Residual analysis (test set):')
    residuals = y_test - y_pred
    print(f"    Mean residual    : {residuals.mean():.2f}")
    print(f"    Std of residuals : {residuals.std():.2f}")
    print(f"    Max error        : {np.max(np.abs(residuals)):.2f}")

    print('\n[7] Subgroup errors (MAE by cohort, test set):')
    subgroup_df = subgroup_errors(best_model, X_test, y_test, data, idx_test)
    print(subgroup_df.to_string(index=False))

    return {
        'best_params': best_params,
        'best_model': best_model,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'validation': val_metrics,
        'test': test_metrics,
        'cv': cv_metrics,
        'comparisons': comparisons,
        'subgroups': subgroup_df,
        'residuals': residuals,
        'y_test': y_test,
        'y_pred': y_pred,
    }
