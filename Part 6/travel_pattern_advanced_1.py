import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Load data
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'

west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')

# Standardize column names for analysis
west_midlands_df = west_midlands_df.rename(columns={
    'How would you rate EV charging station distribution in the West Midlands?': 
        'EV charging station distribution rating',
    'First three digit postcode ': 
        'First three digit postcode',
    'Do you believe involving electric car users in the planning and design process of charging infrastructure would lead to better outcomes?':
        'Involve EV users in planning'
})

newcastle_df = newcastle_df.rename(columns={
    'How would you rate EV charging station distribution in Newcastle?': 
        'EV charging station distribution rating',
    'Do you believe involving EV users in the planning and design process of charging infrastructure would lead to better outcomes?':
        'Involve EV users in planning'
})

# Columns of interest
columns_of_interest = {
    "time_on_road": "How does charging station availability affect your time on the road?",
    "recharge_level": "At what charge level do you typically recharge your EV?",
    "planning_trips": "How do you include charging stations in planning trips over 50 km?",
    "rating_distribution": "EV charging station distribution rating",
    "parking_influence": "How has EV charging availability influenced your parking habits?",
    "parking_convenience": "How convenient is parking near charging stations versus regular spaces?"
}

# Extract and standardize columns
west_midlands_filtered = west_midlands_df[list(columns_of_interest.values())]
newcastle_filtered = newcastle_df[list(columns_of_interest.values())]

west_midlands_filtered.columns = columns_of_interest.keys()
newcastle_filtered.columns = columns_of_interest.keys()

# Add region labels
west_midlands_filtered['region'] = 'West Midlands'
newcastle_filtered['region'] = 'Newcastle'

# Combine datasets
combined_data = pd.concat([west_midlands_filtered, newcastle_filtered], ignore_index=True)

# Handle missing values
combined_data = combined_data.dropna()

# Encode categorical variables
le = LabelEncoder()
for column in combined_data.columns:
    if combined_data[column].dtype == 'object':
        combined_data[column] = le.fit_transform(combined_data[column])

# Prepare data for Machine Learning
X = combined_data.drop(['region', 'time_on_road'], axis=1)  # Features
y = combined_data['time_on_road']  # Target

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create a Keras model
def create_model(learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(y.unique()), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model with KerasClassifier
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameter grid
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100],
    'learning_rate': [0.001, 0.01],
    'dropout_rate': [0.2, 0.3, 0.4]
}

# Perform Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and results
print("Best Parameters:", grid_result.best_params_)
print("Best Accuracy:", grid_result.best_score_)

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Save the best model
best_model.model.save(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\optimized_travel_behaviour_model.h5')

print("Optimized model training complete and saved!")

# Advanced visualizations for model and data
# Confusion Matrix
predictions = np.argmax(best_model.model.predict(X_test), axis=-1)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(X)
tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
tsne_df['Target'] = y
sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Target', palette='deep', data=tsne_df)
plt.title('t-SNE Visualization of Data')
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
pca_results = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
pca_df['Target'] = y
sns.scatterplot(x='PCA1', y='PCA2', hue='Target', palette='deep', data=pca_df)
plt.title('PCA Visualization of Data')
plt.show()
