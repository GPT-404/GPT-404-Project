AI-Enabled Early Warning & Response System for Hilly MSMEs
Overview

This project implements an AI-driven landslide and disaster risk prediction and alerting system designed for Micro, Small, and Medium Enterprises (MSMEs) in hilly regions.
The system combines machine learning (LightGBM) and time-series forecasting (Prophet) to predict hazard risk levels (Low, Moderate, High, Severe) based on environmental factors such as rainfall, soil moisture, and temperature.

Predicted risks are delivered as real-time alerts via Telegram to MSME vendors, enabling them to take timely action and minimize potential losses.

Features
AI/ML-powered forecasting and classification

Early warning alerts based on predicted hazard levels

Telegram integration for instant vendor notifications

Designed for MSMEs in hilly, disaster-prone areas

Modular design, making it easy to integrate with IoT sensor data in the future


Tech Stack
Python

Prophet (rainfall forecasting)

LightGBM (hazard risk classification)

scikit-learn (model evaluation)

Telegram Bot API (vendor messaging)


How It Works
Data generation: Synthetic rainfall, soil moisture, and temperature data are generated (can be replaced with real IoT sensor data).

Forecasting: Prophet predicts the next 24 hours of rainfall.

Feature engineering: Derived features such as rolling rainfall and soil changes are created.

Classification: LightGBM model predicts hazard risk level.

Vendor alerts: Telegram bot sends actionable messages to registered MSME vendors.


Example Alert
Severe: Evacuate immediately. Landslide danger in your area.


Future Improvements
Replace synthetic data with real-time IoT sensor feeds (rain gauges, soil moisture sensors, temperature loggers).

Enable multi-vendor targeting with location-based alerts.

Add a dashboard for authorities to monitor risk patterns.