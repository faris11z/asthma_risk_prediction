# Asthma Risk Prediction using Random Forest Regressor

> **v2.0** — Major upgrade from the original Decision Tree Classifier project by Sulaiman Faris, Jayapriyan, and Hariharan.

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

- **Personal health profiling** — Age, Height, Gender, Smoking, Asthma history
- **Live environmental data** — Real-time Temperature, Humidity, PM2.5, PM10 from Open-Meteo API
- **ML-powered prediction** — Random Forest Regressor trained on 5,000 records
- **Risk assessment** — SAFE / MODERATE / RISK zones based on PEFR ratio
- **Responsive web UI** — Dark theme, field tooltips, city autocomplete, mobile-friendly

---

## Architecture

```
myapp/
├── app.py                 # Flask web server & JSON API
├── weather_data.py        # ML model, weather fetching, prediction logic
├── PEFR_Data_Set.csv      # Training dataset (5,000 rows)
├── PEFR_predictor.joblib  # Serialised trained model
├── templates/
│   └── index.html         # Frontend (single-page app)
├── web/                   # Legacy v1.0 Eel-based frontend
├── requirements.txt       # Python dependencies
└── Asthma Risk Prediction.py  # Legacy v1.0 notebook
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
cd Asthma-Risk-Prediction-using-Decision-Tree/myapp
pip install -r requirements.txt
python3 app.py
```

Open **http://127.0.0.1:5000** in your browser.

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
- **Why not Decision Tree?**: PEFR is a continuous value (regression), not a category. A classifier treats each unique PEFR as a class — conceptually wrong. Also, a single tree overfits; 100 trees in a forest generalise far better.
- **Features**: Age, Height, Gender, Smoking, AsthmaHistory, Temperature, Humidity, PM2.5, PM10
- **Target**: PEFR (Peak Expiratory Flow Rate in L/min)
- **Training data**: 5,000 synthetically generated records based on standard respiratory physiology
- **Hyperparameters**: `n_estimators=100`, `max_depth=15`, `min_samples_leaf=5`

### Performance comparison

| Model | MAE (L/min) | R² | 
|---|---|---|
| DecisionTreeClassifier | 51.2 | 0.50 |
| DecisionTreeRegressor | 40.4 | 0.69 |
| **RandomForestRegressor** | **34.2** | **0.78** |
| GradientBoostingRegressor | 33.6 | 0.79 |

### PEFR reference formula

The synthetic dataset was generated using standard respiratory physiology:

| Factor | Male | Female |
|--------|------|--------|
| Base formula | `3.4 × Ht − 1.6 × Age + 70` | `3.0 × Ht − 1.4 × Age + 30` |
| Smoking penalty | −30 to −80 L/min | −30 to −80 L/min |
| Asthma penalty | −20 to −60 L/min | −20 to −60 L/min |

### Weather data sources

1. **Open-Meteo** (primary) — Free API, no key required. Global coverage. Provides temperature, humidity, PM2.5, PM10.
2. **IQAir** (fallback) — Scraped from IQAir city pages.
3. **Generative fallback** — City-name-seeded realistic defaults if both APIs fail.

---

## Risk Zones

| Zone | Ratio | Meaning |
|------|-------|---------|
| SAFE | ≥80% | Your lung function is within normal range for your profile and environment. |
| MODERATE | 50–79% | Some reduction detected. Monitor symptoms and consider environmental triggers. |
| RISK | <50% | Significant reduction. Seek medical attention and review exposure to pollutants. |

---

## Future Work: Room-Level Monitoring

The current system operates at **city level** — it fetches ambient AQI and weather data for the entered city. This is a coarse approximation because:

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

## Contact

- Sulaiman Faris — sulaiman11faris@gmail.com
- Jayapriyan — jayapriyan11802@gmail.com
