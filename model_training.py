import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'PEFR_Data_Set.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'PEFR_predictor.joblib')

FEATURE_NAMES = ['Age', 'Height', 'Gender', 'Smoking', 'AsthmaHistory',
                 'Outdoor Temperature', 'Humidity', 'PM 2.5', 'PM 10']

TARGET = 'PEFR'

HYPERPARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1,
}

def load_data():
    return pd.read_csv(DATA_PATH)

def _train_model():
    data = load_data()
    X = data[FEATURE_NAMES].values
    y = data[TARGET].values
    model = RandomForestRegressor(**HYPERPARAMS)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"RandomForestRegressor trained on {len(data)} records with {len(FEATURE_NAMES)} features")
    return model

def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return _train_model()

def save_best_model(model, params=None):
    joblib.dump(model, MODEL_PATH)
    print(f"Best model saved to {os.path.basename(MODEL_PATH)}"
          + (f"  {params}" if params else ""))
    return model

def get_model_info():
    data = load_data()
    return {
        'rows': len(data),
        'features': FEATURE_NAMES,
        'target': TARGET,
        'model': 'RandomForestRegressor',
        'hyperparameters': HYPERPARAMS,
        'trained': os.path.exists(MODEL_PATH),
    }

def predict_pefr(features):
    model = get_model()
    return float(model.predict([features])[0])
