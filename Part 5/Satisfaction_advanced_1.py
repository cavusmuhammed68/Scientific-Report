# -*- coding: utf-8 -*-
"""
Updated on Wed Jan  1 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint

# File paths
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')

# Strip column names to remove trailing/leading spaces
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Combine textual suggestions and satisfaction ratings
newcastle_text = newcastle_df['Suggestions for improving EV charging facilities'].fillna('')
west_midlands_text = west_midlands_df['Suggestions for improving EV charging facilities'].fillna('')
all_text = pd.concat([newcastle_text, west_midlands_text], axis=0).fillna('')

newcastle_target = newcastle_df['How would you rate EV charging station distribution in Newcastle?'].fillna(0)
west_midlands_target = west_midlands_df['How would you rate EV charging station distribution in the West Midlands?'].fillna(0)
all_target = pd.concat([newcastle_target, west_midlands_target], axis=0).astype(int)

# Text vectorization with bigrams
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_text = vectorizer.fit_transform(all_text).toarray()

# Add statistical features
word_count = all_text.apply(lambda x: len(x.split()))
sentence_length = all_text.apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)

# Combine features
X = np.hstack((X_text, word_count.values.reshape(-1, 1), sentence_length.values.reshape(-1, 1)))

# Check for NaN values in the final feature matrix
if np.any(np.isnan(X)):
    print("NaN values detected in the feature matrix. Replacing with 0.")
    X = np.nan_to_num(X)

# Dimensionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, all_target, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)

# Hyperparameter tuning for Random Forest
param_dist_rf = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=20, cv=3, scoring='r2', random_state=42)
search_rf.fit(X_train, y_train)
best_rf = search_rf.best_estimator_

# Train XGBoost
xgb_model.fit(X_train, y_train)

# Evaluate Models
models = {
    'Random Forest (Optimized)': best_rf,
    'XGBoost': xgb_model
}
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'R²': r2}

    print(f"Model: {name}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R² Score: {r2:.3f}\n")

# Feature Importance Comparison
rf_importances = best_rf.feature_importances_
xgb_importances = xgb_model.feature_importances_

# Sort features by importance for Random Forest
sorted_indices_rf = rf_importances.argsort()[-10:][::-1]
top_features_rf = [vectorizer.get_feature_names_out()[i] for i in sorted_indices_rf if i < len(vectorizer.get_feature_names_out())]

# Sort features by importance for XGBoost
sorted_indices_xgb = xgb_importances.argsort()[-10:][::-1]
top_features_xgb = [vectorizer.get_feature_names_out()[i] for i in sorted_indices_xgb if i < len(vectorizer.get_feature_names_out())]

# Create a comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Random Forest Feature Importance
axes[0].barh(top_features_rf, rf_importances[sorted_indices_rf], color='skyblue')
axes[0].set_xlabel('Importance', fontsize=14)
axes[0].set_ylabel('Feature', fontsize=14)
axes[0].set_title('Top Features (Random Forest)', fontsize=16)

# XGBoost Feature Importance
axes[1].barh(top_features_xgb, xgb_importances[sorted_indices_xgb], color='orange')
axes[1].set_xlabel('Importance', fontsize=14)
axes[1].set_ylabel('Feature', fontsize=14)
axes[1].set_title('Top Features (XGBoost)', fontsize=16)

# Adjust layout and show
plt.tight_layout()
plt.show()
