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
from catboost import CatBoostRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
import shap

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

# Generate feature names
tfidf_feature_names = vectorizer.get_feature_names_out()
statistical_feature_names = ['word_count', 'sentence_length']
feature_names = list(tfidf_feature_names) + statistical_feature_names

# Check for NaN values in the final feature matrix
if np.any(np.isnan(X)):
    print("NaN values detected in the feature matrix. Replacing with 0.")
    X = np.nan_to_num(X)

# Dimensionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=50, random_state=42)
X_reduced = svd.fit_transform(X)

# Map components to their single most influential feature
components = svd.components_
single_feature_mapping = []
for i, component in enumerate(components):
    top_index = np.argmax(np.abs(component))  # Index of the most influential feature
    top_feature = feature_names[top_index]  # Get the corresponding feature name
    single_feature_mapping.append(top_feature)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, all_target, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
catboost_model = CatBoostRegressor(random_state=42, verbose=0)

# Hyperparameter tuning for Random Forest
param_dist_rf = {
    'n_estimators': randint(200, 1000),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=50, cv=5, scoring='r2', random_state=42)
search_rf.fit(X_train, y_train)
best_rf = search_rf.best_estimator_

# Train other models
xgb_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

# Evaluate Models
models = {
    'Random Forest (Optimized)': best_rf,
    'XGBoost': xgb_model,
    'CatBoost': catboost_model
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

# SHAP Analysis for Random Forest
explainer_rf = shap.Explainer(best_rf, X_train)
shap_values_rf = explainer_rf(X_test)

# SHAP Analysis for CatBoost
explainer_cat = shap.Explainer(catboost_model, X_train)
shap_values_cat = explainer_cat(X_test)

# Use single-word mapping for SHAP visualization
shap.summary_plot(shap_values_rf, X_test, feature_names=single_feature_mapping)
shap.summary_plot(shap_values_cat, X_test, feature_names=single_feature_mapping)

# SHAP Force Plot for Random Forest (First Prediction)
shap.force_plot(explainer_rf.expected_value, shap_values_rf[0].values, X_test[0], matplotlib=True, feature_names=single_feature_mapping)

# SHAP Force Plot for CatBoost (First Prediction)
shap.force_plot(explainer_cat.expected_value, shap_values_cat[0].values, X_test[0], matplotlib=True, feature_names=single_feature_mapping)
