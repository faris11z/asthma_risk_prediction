---
title: Asthma Risk Prediction
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.40.0"
app_file: app.py
pinned: false
---

# Asthma Risk Prediction using Random Forest Regressor

> **v2.0** -- Major upgrade from the original Decision Tree Classifier project by Sulaiman Faris.

A web-based application that predicts asthma risk by comparing your actual Peak Expiratory Flow Rate (PEFR) against a predicted healthy baseline for your demographic profile and current environmental conditions.

---

## v2.0 Changelog

| Area | v1.0 (original) | v2.0 (current) |
|---|---|---|
| **Dataset** | 54 rows | 5,000 rows |
| **Model** | DecisionTreeClassifier | RandomForestRegressor (100 trees) |
| **Model accuracy** | R² = 0.50, MAE = 51.2 L/min | R² = 0.78, MAE = 34.2 L/min |
| **Input features** | Gender only | Age, Height, Gender, Smoking, Asthma History |
| **API** | Form-based (Flask templates) | JSON API (`/api/predict`) |
| **Frontend** | Broken multi-form layout | Single-page app, dark theme, tooltips, responsive |
| **Weather source** | IQAir only (India) | Open-Meteo (global) + IQAir + fallback |
| **City scope** | Tamil Nadu only | Worldwide (60+ cities pre-configured) |

---

## Features

- **Personal health profiling** -- Age, Height, Gender, Smoking, Asthma history
- **Live environmental data** -- Real-time Temperature, Humidity, PM2.5, PM10 from Open-Meteo API
- **ML-powered prediction** -- Random Forest Regressor trained on 5,000 records
- **Risk assessment** -- SAFE / MODERATE / RISK zones based on PEFR ratio
- **Responsive web UI** -- Dark theme, field tooltips, city autocomplete, mobile-friendly

---

## Architecture

```
myapp/
├── app.py                  # Flask web server & JSON API (orchestration)
├── weather_data.py         # Weather fetching only (Open-Meteo / IQAir / fallback)
├── model_training.py       # Random Forest training, caching, prediction helper
├── model_evaluation.py     # Metrics: holdout, 5-fold CV, subgroup errors
├── model_visualization.py  # Matplotlib plots → eval_plots/
├── evaluate.py             # Runnable: retrain + evaluate + visualize (python3 evaluate.py)
├── PEFR_Data_Set.csv       # Training dataset (5,000 rows)
├── PEFR_predictor.joblib   # Serialised trained model
├── eval_plots/             # Generated evaluation charts
├── templates/
│   └── index.html          # Frontend (single-page app)
├── web/                    # Legacy v1.0 Eel-based frontend
└── requirements.txt        # Python dependencies
```

### Data flow

```
User inputs (age, height, gender, smoking, asthma, actual PEFR, city)
       │
       ▼
Flask API  ──►  weather_data.py  ──►  Weather API (Open-Meteo / IQAir / fallback)
       │                                      │
       │                                      ▼
       │                              Temperature, Humidity, PM2.5, PM10
       │                                      │
       ▼                                      │
   Combine features ──────────────────────────┘
       │
       ▼
   Random Forest Regressor ──►  Predicted PEFR
       │
       ▼
   Compare actual vs predicted PEFR ──►  Ratio → SAFE (≥80%) / MODERATE (≥50%) / RISK (<50%)
```

---

## Installation

```bash
git clone <repo-url>
cd asthma_risk_prediction/myapp
pip install -r requirements.txt
python3 app.py
```

Open **http://127.0.0.1:7860** in your browser.

---

## API

### `POST /api/predict`

```json
{
  "city": "chennai",
  "age": "30",
  "height": "170",
  "gender": "1",
  "smoking": "0",
  "asthma": "0",
  "actual_pefr": "500"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "city": "chennai",
    "user": {
      "age": 30, "height": 170, "gender": "Male",
      "smoking": "No", "asthma_history": "No", "actual_pefr": 500
    },
    "environment": {
      "temperature": 32.5, "humidity": 65, "pm25": 18.2, "pm10": 30.4
    },
    "result": {
      "predicted_pefr": 540, "ratio": 92.6, "zone": "SAFE"
    }
  }
}
```

### `GET /api/model-info`

Returns dataset row count, feature names, target column, and training status.

---

## Model

- **Algorithm**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- **Why not Decision Tree?**: PEFR is a continuous value (regression), not a category. A classifier treats each unique PEFR as a class -- conceptually wrong. Also, a single tree overfits; 100 trees in a forest generalise far better.
- **Features**: Age, Height, Gender, Smoking, AsthmaHistory, Temperature, Humidity, PM2.5, PM10
- **Target**: PEFR (Peak Expiratory Flow Rate in L/min)
- **Training data**: 5,000 synthetically generated records based on standard respiratory physiology
- **Hyperparameters**: tuned via `GridSearchCV` -- `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `min_samples_split=2`
- **Evaluation**: run `python3 evaluate.py` (retrains fresh, retunes hyperparameters, regenerates all reports & plots). Add `--fast` for a quick run with a smaller grid.

### Evaluation pipeline: train / validation / test split

```
Dataset (5,000)
   ├── Train (70% = 3,500)      ──► GridSearchCV (5-fold) finds best hyperparameters
   ├── Validation (15% = 750)   ──► sanity check against overfitting
   └── Test (15% = 750, unused during tuning)  ──► final honest metrics
```

### Performance comparison

| Model | MAE (L/min) | R² | 
|---|---|---|
| DecisionTreeRegressor | 40.61 | 0.69 |
| RandomForestRegressor (default) | 34.25 | 0.78 |
| **RandomForestRegressor (tuned)** | **34.00** | **0.78** |
| GradientBoostingRegressor | 33.64 | 0.78 |

### Evaluation metrics

| Metric | Test set (untouched) | 5-fold CV (best params) |
|---|---|---|
| MAE | 34.93 L/min | 34.00 ± 0.63 L/min |
| RMSE | 44.32 L/min | 42.73 ± 0.90 L/min |
| R² | 0.77 | 0.7786 ± 0.0119 |
| MAPE | 7.19% | -- |
| Explained Variance | 0.77 | -- |

Overfitting check: validation MAE 33.73 vs CV-train MAE 34.02 → gap 0.29 L/min, no overfitting.

### Feature importance

| Feature | Importance |
|---|---|
| Gender | 0.678 |
| Height | 0.096 |
| Age | 0.068 |
| Smoking | 0.061 |
| Temperature | 0.024 |
| Humidity | 0.022 |
| PM2.5 | 0.017 |
| PM10 | 0.017 |
| Asthma History | 0.017 |

### Key trends (from evaluation)

- **Tuning paid off**: GridSearchCV (36 combos × 5 folds) improved CV MAE from 34.25 → 34.00 L/min (best: `max_depth=10`, 200 trees).
- **Gender dominates** (68% importance) -- expected, males average 580 vs 439 L/min for females in the dataset.
- **Physiological factors** (Gender + Height + Age ≈ 84%) drive the prediction far more than environmental ones -- consistent with medical literature where PEFR depends primarily on age, height and sex.
- **Asthmatics are harder to predict**: MAE 40.3 vs 33.8 L/min for non-asthmatics -- a larger asthmatic cohort would improve this.
- **Mid-PEFR band is the hardest to predict** (MAE 38.5), while high-PEFR is easiest (30.8) -- regression toward the mean in the middle range.
- **Mean residual ≈ −1.1 L/min** → essentially zero systematic bias; errors are random.

### Visualizations (`eval_plots/`)

| Plot | What it shows |
|---|---|
| `feature_importance.png` | Horizontal bar chart ranking the 9 features |
| `pred_vs_actual.png` | Scatter of predictions hugging the y=x line |
| `residuals.png` | Residual histogram + heteroscedasticity check |

### PEFR reference formula

The synthetic dataset was generated using standard respiratory physiology:

| Factor | Male | Female |
|--------|------|--------|
| Base formula | `3.4 × Ht − 1.6 × Age + 70` | `3.0 × Ht − 1.4 × Age + 30` |
| Smoking penalty | −30 to −80 L/min | −30 to −80 L/min |
| Asthma penalty | −20 to −60 L/min | −20 to −60 L/min |

### Weather data sources

1. **Open-Meteo** (primary) -- Free API, no key required. Global coverage. Provides temperature, humidity, PM2.5, PM10.
2. **IQAir** (fallback) -- Scraped from IQAir city pages.
3. **Generative fallback** -- City-name-seeded realistic defaults if both APIs fail.

---

## Risk Zones

| Zone | Ratio | Meaning |
|------|-------|---------|
| SAFE | ≥80% | Your lung function is within normal range for your profile and environment. |
| MODERATE | 50–79% | Some reduction detected. Monitor symptoms and consider environmental triggers. |
| RISK | <50% | Significant reduction. Seek medical attention and review exposure to pollutants. |

---

## Future Work: Room-Level Monitoring

The current system operates at **city level** -- it fetches ambient AQI and weather data for the entered city. This is a coarse approximation because:

- Indoor air quality can differ dramatically from outdoor readings
- Individual rooms have unique PM, humidity, and temperature profiles
- People spend ~90% of their time indoors

### Planned enhancements

| Feature | Approach |
|---------|----------|
| **IoT sensor integration** | Deploy low-cost PM + temp/humidity sensors (e.g., PMS5003 + DHT22 + ESP32/ESP8266) in individual rooms |
| **Real-time room telemetry** | Sensors push data via MQTT/HTTP to the backend every 5–15 minutes |
| **Room-level dashboard** | Web UI showing per-room AQI, trends, and personalised risk |
| **Threshold alerts** | Push/email notifications when room conditions cross user-defined safety thresholds |
| **Historical analytics** | Identify patterns linking room conditions to PEFR changes over time |

### Sensor stack (proposed)

```
Room  ──►  PMS5003 (PM2.5/PM10)
       ──►  DHT22   (Temp/Humidity)
       ──►  ESP32   (WiFi + MQTT)
                │
                ▼
          MQTT Broker ──►  Flask Backend ──►  Database (SQLite/PostgreSQL)
                                       │
                                       ▼
                                 Web Dashboard
```

This transforms the app from a **city-level screener** into a **personal indoor environment health monitor**, enabling proactive asthma management at the room level.

---

## License

MIT
