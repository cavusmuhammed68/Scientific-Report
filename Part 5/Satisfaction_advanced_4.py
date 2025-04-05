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
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

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

# Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_text))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Common Terms in Suggestions', fontsize=16)

output_path = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 5\wordcloud_600dpi.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
plt.show()

# Text vectorization with bigrams
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
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

# Feature selection
selector = SelectKBest(score_func=f_regression, k=min(100, X.shape[1]))
X_selected = selector.fit_transform(X, all_target)
selected_feature_names = [feature_names[i] for i in selector.get_support(indices=True)]

# Dimensionality reduction with Truncated SVD
svd = TruncatedSVD(n_components=min(50, X_selected.shape[1]), random_state=42)
X_reduced = svd.fit_transform(X_selected)

# Map components to their single most influential feature
components = svd.components_
single_feature_mapping = []
for i, component in enumerate(components):
    top_index = np.argmax(np.abs(component))  # Index of the most influential feature
    top_feature = selected_feature_names[top_index]  # Get the corresponding feature name
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
    'n_estimators': randint(300, 1200),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=100, cv=5, scoring='r2', random_state=42)
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

# Round SHAP values and feature values to two decimals
shap_values_rf_rounded = shap_values_rf.values.round(2)
shap_values_cat_rounded = shap_values_cat.values.round(2)
X_test_rounded = X_test.round(2)

# Use single-word mapping for SHAP visualization
shap.summary_plot(shap_values_rf, X_test, feature_names=single_feature_mapping)
shap.summary_plot(shap_values_cat, X_test, feature_names=single_feature_mapping)

# SHAP Force Plot for Random Forest (First Prediction with Two Decimal Places)
shap.force_plot(
    explainer_rf.expected_value,
    shap_values_rf_rounded[0],  # Use rounded SHAP values
    X_test_rounded[0],          # Use rounded feature values
    matplotlib=True,
    feature_names=single_feature_mapping
)

# SHAP Force Plot for CatBoost (First Prediction with Two Decimal Places)
shap.force_plot(
    explainer_cat.expected_value,
    shap_values_cat_rounded[0],  # Use rounded SHAP values
    X_test_rounded[0],           # Use rounded feature values
    matplotlib=True,
    feature_names=single_feature_mapping
)






import matplotlib.pyplot as plt

# Adjust settings for SHAP summary plots
plt.rcParams.update({
    'savefig.dpi': 600,           # Set DPI for saved figures
    'font.size': 16,              # General font size for titles and labels
    'axes.labelsize': 14,         # Font size for x and y labels
    'xtick.labelsize': 14,        # Font size for x-axis tick labels
    'ytick.labelsize': 14         # Font size for y-axis tick labels
})

# Save SHAP summary plot for Random Forest
shap.summary_plot(shap_values_rf, X_test, feature_names=single_feature_mapping, show=False)
plt.title('SHAP Summary Plot - Random Forest', fontsize=16)
plt.savefig("shap_summary_rf.png", dpi=600, bbox_inches='tight')
plt.clf()  # Clear figure

# Save SHAP summary plot for CatBoost
shap.summary_plot(shap_values_cat, X_test, feature_names=single_feature_mapping, show=False)
plt.title('SHAP Summary Plot - CatBoost', fontsize=16)
plt.savefig("shap_summary_catboost.png", dpi=600, bbox_inches='tight')
plt.clf()

# Save SHAP force plot for Random Forest
shap.force_plot(
    explainer_rf.expected_value,
    shap_values_rf_rounded[0],
    X_test_rounded[0],
    matplotlib=True,
    feature_names=single_feature_mapping
)
plt.title('SHAP Force Plot - Random Forest', fontsize=16)
plt.savefig("shap_force_rf.png", dpi=600, bbox_inches='tight')
plt.clf()

# Save SHAP force plot for CatBoost
shap.force_plot(
    explainer_cat.expected_value,
    shap_values_cat_rounded[0],
    X_test_rounded[0],
    matplotlib=True,
    feature_names=single_feature_mapping
)
plt.title('SHAP Force Plot - CatBoost', fontsize=16)
plt.savefig("shap_force_catboost.png", dpi=600, bbox_inches='tight')
plt.clf()















