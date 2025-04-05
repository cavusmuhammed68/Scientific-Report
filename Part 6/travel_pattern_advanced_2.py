# Part 1: Data Preparation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data files
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'

# Read datasets
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')

# Standardize column names
columns_map_west_midlands = {
    'How would you rate EV charging station distribution in the West Midlands?': 'EV charging station distribution rating',
    'First three digit postcode ': 'First three digit postcode',
    'Do you believe involving electric car users in the planning and design process of charging infrastructure would lead to better outcomes?': 'Involve EV users in planning'
}

columns_map_newcastle = {
    'How would you rate EV charging station distribution in Newcastle?': 'EV charging station distribution rating',
    'Do you believe involving EV users in the planning and design process of charging infrastructure would lead to better outcomes?': 'Involve EV users in planning'
}

west_midlands_df.rename(columns=columns_map_west_midlands, inplace=True)
newcastle_df.rename(columns=columns_map_newcastle, inplace=True)

# Columns of interest
columns_of_interest = {
    "time_on_road": "How does charging station availability affect your time on the road?",
    "recharge_level": "At what charge level do you typically recharge your EV?",
    "planning_trips": "How do you include charging stations in planning trips over 50 km?",
    "rating_distribution": "EV charging station distribution rating",
    "parking_influence": "How has EV charging availability influenced your parking habits?",
    "parking_convenience": "How convenient is parking near charging stations versus regular spaces?"
}

# Filter and standardize columns
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
combined_data.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
for column in combined_data.columns:
    if combined_data[column].dtype == 'object':
        combined_data[column] = le.fit_transform(combined_data[column])

# Define features and target
X = combined_data.drop(['region', 'time_on_road'], axis=1)
y = combined_data['time_on_road']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set path for saving figures
figures_path = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 6'

# Part 2: Model Definition and Training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define function to create the model
def create_model(learning_rate=0.001, dropout_rate=0.3):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate and train the model
model = create_model()
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save(f"{figures_path}/trained_model.h5")
print("Model training complete and saved.")

# Part 3: Model Evaluation and Visualizations
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Plot training history
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/training_history.png")
    plt.show()

# Call the function to plot history
plot_training_history(history, figures_path)

# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()

# Predict and evaluate
y_pred = np.argmax(model.predict(X_test), axis=1)
plot_confusion_matrix(y_test, y_pred, classes=np.unique(y_train), save_path=figures_path)

# Classification Report
report = classification_report(y_test, y_pred, target_names=[str(label) for label in np.unique(y_train)])
print("Classification Report:")
print(report)

# Part 4: Advanced Dimensionality Reduction and Visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# t-SNE Visualization
def plot_tsne(X, y, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Target', palette='deep', data=tsne_df)
    plt.title('t-SNE Visualization of Data')
    plt.savefig(f"{save_path}/tsne_visualization.png")
    plt.show()

plot_tsne(X, y, figures_path)

# PCA Visualization
def plot_pca(X, y, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Target', palette='deep', data=pca_df)
    plt.title('PCA Visualization of Data')
    plt.savefig(f"{save_path}/pca_visualization.png")
    plt.show()

plot_pca(X, y, figures_path)

# Part 5: Hyperparameter Tuning with Grid Search
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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

# Save the optimized model
best_model = grid_result.best_estimator_
best_model.model.save(f"{figures_path}/optimized_model.h5")
print("Optimized model training complete and saved!")
