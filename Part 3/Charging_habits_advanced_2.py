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
    west_midlands_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlandss.xlsx')
    newcastle_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastlee.xlsx')

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
        data["User Patience^2"] = data["User Patience (Minutes)"] ** 2
        data["Patience x Hour"] = data["User Patience (Minutes)"] * data["Charging Time (Hour)"]
    return data

# Preprocess the data and handle empty datasets
west_midlands_processed = preprocess_and_engineer(west_midlands_data_cleaned)
west_midlands_processed = handle_empty_data(west_midlands_processed, "West Midlands")

newcastle_processed = preprocess_and_engineer(newcastle_data_cleaned)
newcastle_processed = handle_empty_data(newcastle_processed, "Newcastle")

def train_and_evaluate(data, region_name):
    # Prepare features and target
    X = data[["User Patience (Minutes)", "User Patience^2", "Patience x Hour"]]
    y = data["Charging Time (Hour)"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build Neural Network
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return history, y_test, y_pred, mse, r2


# Train and evaluate for each region
history_wm, y_test_wm, y_pred_wm, mse_wm, r2_wm = train_and_evaluate(west_midlands_processed, "West Midlands")
history_ncl, y_test_ncl, y_pred_ncl, mse_ncl, r2_ncl = train_and_evaluate(newcastle_processed, "Newcastle")

# Create a 2x2 grid for plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)

# Plot 1: Loss Curves for West Midlands
axes[0, 0].plot(history_wm.history['loss'], label='Training Loss')
axes[0, 0].plot(history_wm.history['val_loss'], label='Validation Loss')
axes[0, 0].set_xlabel('Epochs', fontsize=16)
axes[0, 0].set_ylabel('Mean Squared Error', fontsize=16)
axes[0, 0].tick_params(axis='x', labelsize=14)
axes[0, 0].tick_params(axis='y', labelsize=14)
axes[0, 0].set_title('Loss Curves (West Midlands)', fontsize=16)
axes[0, 0].legend(fontsize=14)
axes[0, 0].grid(True)

# Plot 2: Predicted vs Actual for West Midlands
axes[0, 1].scatter(y_test_wm, y_pred_wm, color='orange', label="Predicted vs Actual")
axes[0, 1].plot([min(y_test_wm), max(y_test_wm)], [min(y_test_wm), max(y_test_wm)], color='blue', linestyle='--', label="Ideal Fit")
axes[0, 1].set_xlabel("Actual Charging Time (Hour)", fontsize=16)
axes[0, 1].set_ylabel("Predicted Charging Time (Hour)", fontsize=16)
axes[0, 1].tick_params(axis='x', labelsize=14)
axes[0, 1].tick_params(axis='y', labelsize=14)
axes[0, 1].set_title("Predicted vs Actual Charging Times (West Midlands)", fontsize=16)
axes[0, 1].legend(fontsize=14)
axes[0, 1].grid(True)

# Plot 3: Loss Curves for Newcastle
axes[1, 0].plot(history_ncl.history['loss'], label='Training Loss')
axes[1, 0].plot(history_ncl.history['val_loss'], label='Validation Loss')
axes[1, 0].set_xlabel('Epochs', fontsize=16)
axes[1, 0].set_ylabel('Mean Squared Error', fontsize=16)
axes[1, 0].tick_params(axis='x', labelsize=14)
axes[1, 0].tick_params(axis='y', labelsize=14)
axes[1, 0].set_title('Loss Curves (North East)', fontsize=16)
axes[1, 0].legend(fontsize=14)
axes[1, 0].grid(True)

# Plot 4: Predicted vs Actual for Newcastle
axes[1, 1].scatter(y_test_ncl, y_pred_ncl, color='orange', label="Predicted vs Actual")
axes[1, 1].plot([min(y_test_ncl), max(y_test_ncl)], [min(y_test_ncl), max(y_test_ncl)], color='blue', linestyle='--', label="Ideal Fit")
axes[1, 1].set_xlabel("Actual Charging Time (Hour)", fontsize=16)
axes[1, 1].set_ylabel("Predicted Charging Time (Hour)", fontsize=16)
axes[1, 1].tick_params(axis='x', labelsize=14)
axes[1, 1].tick_params(axis='y', labelsize=14)
axes[1, 1].set_title("Predicted vs Actual Charging Times (North East)", fontsize=16)
axes[1, 1].legend(fontsize=14)
axes[1, 1].grid(True)

# Save the plot as a high-resolution image
save_path = r'C:/Users/cavus/Desktop/Dilum_Ekip_Paper/New Study/Loss_Curves_and_Predicted_vs_Actual.png'
plt.savefig(save_path, dpi=600)

# Display the plots
plt.show()

