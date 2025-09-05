!pip install prophet lightgbm scikit-learn matplotlib seaborn requests python-dotenv

import os
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
VENDORS = [int(x) for x in os.getenv("VENDOR_IDS", "").split(",")]
BASE_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

dates = pd.date_range(start="2023-01-01", periods=1000, freq="H")
rainfall = np.random.gamma(2, 2, size=len(dates))
soil_moisture = np.random.normal(50, 10, size=len(dates)) + rainfall * 0.5
temperature = np.random.normal(25, 5, size=len(dates))

def assign_risk(r, s, t):
    score = r*0.6 + s*0.3 - (t-25)*0.1
    if score < 20:
        return 0
    elif score < 40:
        return 1
    elif score < 60:
        return 2
    else:
        return 3

risk_label = [assign_risk(r, s, t) for r, s, t in zip(rainfall, soil_moisture, temperature)]

df = pd.DataFrame({
    "date": dates,
    "rainfall": rainfall,
    "soil": soil_moisture,
    "temp": temperature,
    "risk_label": risk_label
})

rainfall_df = df[["date", "rainfall"]].rename(columns={"date": "ds", "rainfall": "y"})
m = Prophet(daily_seasonality=True, weekly_seasonality=True)
m.fit(rainfall_df)
future = m.make_future_dataframe(periods=24, freq="H")
forecast = m.predict(future)

df = df.merge(forecast[["ds", "yhat"]], left_on="date", right_on="ds", how="left")
df.rename(columns={"yhat": "rainfall_forecast"}, inplace=True)

df["rain_3h"] = df["rainfall"].rolling(3).sum()
df["soil_change"] = df["soil"].diff()
df["day_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
df["day_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
df = df.dropna()

features = ["rainfall", "soil", "temp", "rain_3h", "soil_change", "day_sin", "day_cos", "rainfall_forecast"]
X = df[features]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
train_set = lgb.Dataset(X_train, label=y_train)
test_set = lgb.Dataset(X_test, label=y_test)

params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "verbose": -1
}

callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
model = lgb.train(params, train_set, valid_sets=[test_set], num_boost_round=200, callbacks=callbacks)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

unique_labels = np.unique(y_test)
filtered_target_names = [name for i, name in enumerate(["Low", "Moderate", "High", "Severe"]) if i in unique_labels]
print(classification_report(y_test, y_pred_classes, target_names=filtered_target_names, labels=unique_labels))

def send_alert(chat_id, risk_level):
    messages = {
        0: "SAFE: No immediate landslide risk. Continue normal operations.",
        1: "MODERATE: Be cautious. Move essential goods to safer place.",
        2: "HIGH: Landslide risk likely. Secure your stock and avoid travel.",
        3: "SEVERE: Evacuate immediately! Landslide danger in your area."
    }
    resp = requests.post(BASE_URL, data={"chat_id": chat_id, "text": messages[risk_level]})
    print("Status:", resp.status_code, "Response:", resp.json())

if VENDORS:
    latest_risk = y_pred_classes[-1]
    for vendor in VENDORS:
        send_alert(vendor, latest_risk)
    print("Alerts sent to MSME vendors via Telegram")
else:
    print("No vendor chat IDs found. Skipping Telegram alerts")
