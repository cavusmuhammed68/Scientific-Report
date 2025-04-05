# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:45:34 2024

@author: cavus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and preprocess data
try:
    west_midlands_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx')
    newcastle_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx')

    # Parse the sheets
    west_midlands_df = west_midlands_data.parse('Form Responses 1')
    newcastle_df = newcastle_data.parse('Form Responses 1')

    # Extract relevant columns
    charging_time_column = "At what time of day do you usually start charging your EV?"
    user_patience_column = "Given there’s a charging point nearby but it’s occupied by another vehicle, what is the maximum timing you’re willing to wait for your turn?"

    # Clean West Midlands data
    west_midlands_df[charging_time_column] = pd.to_datetime(west_midlands_df[charging_time_column], errors='coerce').dt.hour
    west_midlands_df[user_patience_column] = west_midlands_df[user_patience_column].str.extract('(\d+)').astype(float)
    west_midlands_data_cleaned = west_midlands_df[[charging_time_column, user_patience_column]].dropna()

    # Clean Newcastle data
    newcastle_df[charging_time_column] = pd.to_datetime(newcastle_df[charging_time_column], errors='coerce').dt.hour
    newcastle_df[user_patience_column] = newcastle_df[user_patience_column].str.extract('(\d+)').astype(float)
    newcastle_data_cleaned = newcastle_df[[charging_time_column, user_patience_column]].dropna()
except Exception as e:
    print(f"Error loading or preprocessing data: {e}")

# Function to preprocess and feature engineer
def preprocess_and_engineer(data):
    data = data.rename(columns={
        charging_time_column: "Charging Time (Hour)",
        user_patience_column: "User Patience (Minutes)"
    })
    data["User Patience^2"] = data["User Patience (Minutes)"] ** 2
    data["Patience x Hour"] = data["User Patience (Minutes)"] * data["Charging Time (Hour)"]
    data["Log User Patience"] = np.log1p(data["User Patience (Minutes)"])
    data["Log Charging Time"] = np.log1p(data["Charging Time (Hour)"])
    return data

# Generate synthetic data if dataset is empty
def handle_empty_data(data, region_name):
    if data.empty:
        print(f"{region_name} dataset is empty. Generating synthetic data.")
        np.random.seed(42)
        synthetic_patience = np.random.normal(loc=10, scale=3, size=100)  # Simulated max wait times
        synthetic_charging_time = np.random.normal(loc=14, scale=4, size=100)  # Simulated charging times (hours)

        # Clamp values to realistic ranges
        synthetic_patience = np.clip(synthetic_patience, 1, 30)
        synthetic_charging_time = np.clip(synthetic_charging_time, 0, 23)

        data = pd.DataFrame({
            "User Patience (Minutes)": synthetic_patience,
            "Charging Time (Hour)": synthetic_charging_time
        })
        data = preprocess_and_engineer(data)
    return data

# Preprocess the data and handle empty datasets
west_midlands_processed = preprocess_and_engineer(west_midlands_data_cleaned)
west_midlands_processed = handle_empty_data(west_midlands_processed, "West Midlands")

newcastle_processed = preprocess_and_engineer(newcastle_data_cleaned)
newcastle_processed = handle_empty_data(newcastle_processed, "Newcastle")

# Function to train and evaluate the model
def train_and_evaluate(data, region_name):
    # Prepare features and target
    X = data[["User Patience (Minutes)", "User Patience^2", "Patience x Hour", "Log User Patience"]]
    y = data["Charging Time (Hour)"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build Neural Network
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{region_name} - Test Mean Squared Error: {mse:.4f}")
    print(f"{region_name} - Test R-squared: {r2:.4f}")

    # Visualization: Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Loss Curves for {region_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization: Predicted vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='orange', label="Predicted vs Actual")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', label="Ideal Fit")
    plt.xlabel("Actual Charging Time (Hour)")
    plt.ylabel("Predicted Charging Time (Hour)")
    plt.title(f"Predicted vs Actual Charging Times ({region_name})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualization: Residual Plot
    residuals = y_test - y_pred.flatten()
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='purple', alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Charging Time (Hour)")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot ({region_name})")
    plt.grid(True)
    plt.show()

    # Visualization: Distribution of Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.hist(y_test, bins=20, alpha=0.5, label='Actual', color='blue')
    plt.hist(y_pred.flatten(), bins=20, alpha=0.5, label='Predicted', color='orange')
    plt.xlabel('Charging Time (Hour)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Actual vs Predicted ({region_name})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Train and evaluate for each region
train_and_evaluate(west_midlands_processed, "West Midlands")
train_and_evaluate(newcastle_processed, "Newcastle")
