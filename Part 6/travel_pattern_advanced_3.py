# Part 1: Data Preparation and Loading
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data files
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlandss.xlsx'
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastlee.xlsx'

west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')

rename_dict = {
"First three digit postcode ": "First three digit postcode",
"How do you rate the current distribution of  charging stations along major routes in West Midlands?":
"How do you rate the current distribution of EV charging stations along major routes?",
"How do you rate the current distribution of EV charging stations along major routes in Newcastle?":
"How do you rate the current distribution of EV charging stations along major routes?",
"How do you think  charging facilities could be improved to better meet the needs of electric car users?":
"How do you think EV charging facilities could be improved to better meet the needs of EV users?",
"Do you believe involving electric car users in the planning and design process of charging infrastructure would lead to better outcomes?":
"Do you believe involving EV users in the planning and design process of charging infrastructure would lead to better outcomes?",
"At what state of charge do you typically decide to recharge your electric car?":
"At what state of charge do you typically decide to recharge your EV?",
"How has the availability of  charging stations affected your parking habits?":
"How has the availability of EV charging stations affected your parking habits?"
}

west_midlands_df.rename(columns=rename_dict, inplace=True)
newcastle_df.rename(columns=rename_dict, inplace=True)

columns_of_interest = {
"time_on_road": "How do you think the availability of charging stations affects the time you spend on the road?",
"recharge_level": "At what state of charge do you typically decide to recharge your EV?",
"planning_trips": "When planning longer trips (over 50 km), how do you incorporate charging station locations into your route planning?",
"rating_distribution": "How do you rate the current distribution of EV charging stations along major routes?",
"parking_influence": "How has the availability of EV charging stations affected your parking habits?",
"parking_convenience": "How convenient do you find parking near available charging stations compared to regular parking spots?"
}
# Filter and standardize columns
west_midlands_filtered = west_midlands_df[list(columns_of_interest.values())]
newcastle_filtered = newcastle_df[list(columns_of_interest.values())]

west_midlands_filtered.columns = columns_of_interest.keys()
newcastle_filtered.columns = columns_of_interest.keys()

# Add region labels
west_midlands_filtered['region'] = 'West Midlands'
newcastle_filtered['region'] = 'North East'

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

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Part 2: Advanced Data Splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Part 3: Improved Model Definition

def create_improved_model(learning_rate=0.001, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01):
    input_shape = X_train.shape[1]
    model = Sequential([
        Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Instantiate the improved model
improved_model = create_improved_model(learning_rate=0.001, dropout_rate=0.3, l1_reg=0.01, l2_reg=0.01)

# Add callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)

# Train the improved model
history = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Part 4: Model Evaluation
# Evaluate the improved model
test_loss, test_accuracy = improved_model.evaluate(X_test, y_test, verbose=0)
print(f"Improved Test Accuracy: {test_accuracy:.2f}")

# Predict probabilities
y_pred_proba = improved_model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Compute regression metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

# Save the improved model
figures_path = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 6'
improved_model.save(f"{figures_path}/improved_model.h5")

# Part 5: Model Performance Visualization
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

plot_training_history(history, figures_path)

# Generate classification report
from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.show()

plot_confusion_matrix(y_test, y_pred, figures_path)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(X, y, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Target', palette='deep', data=tsne_df)
    #plt.title('t-SNE Visualization', fontsize=16)
    plt.xlabel('t-SNE1', fontsize=16)
    plt.ylabel('t-SNE2', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"{save_path}/tsne_visualization.png")
    plt.show()

plot_tsne(X_resampled, y_resampled, figures_path)

def plot_pca(X, y, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Target', palette='deep', data=pca_df)
    #plt.title('PCA Visualization', fontsize=16)
    plt.xlabel('PCA1', fontsize=16)
    plt.ylabel('PCA2', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"{save_path}/pca_visualization.png")
    plt.show()

plot_pca(X_resampled, y_resampled, figures_path)










# Part 5: Model Performance Visualization
def plot_training_history(history, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Training History', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.savefig(f"{save_path}/training_history.png", dpi=600, facecolor='w')
    plt.show()

plot_training_history(history, figures_path)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"fontsize": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=16)
    plt.ylabel('True Labels', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=600, facecolor='w')
    plt.show()

plot_confusion_matrix(y_test, y_pred, figures_path)

# Part 6: Dimensionality Reduction Visualization
def plot_tsne(X, y, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Target', palette='deep', data=tsne_df)
    plt.title('t-SNE Visualization', fontsize=16)
    plt.xlabel('t-SNE1', fontsize=16)
    plt.ylabel('t-SNE2', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{save_path}/tsne_visualization.png", dpi=600, facecolor='w')
    plt.show()

plot_tsne(X_resampled, y_resampled, figures_path)

def plot_pca(X, y, save_path):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(X)
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df['Target'] = y
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Target', palette='deep', data=pca_df)
    plt.title('PCA Visualization', fontsize=16)
    plt.xlabel('PCA1', fontsize=16)
    plt.ylabel('PCA2', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(f"{save_path}/pca_visualization.png", dpi=600, facecolor='w')
    plt.show()

plot_pca(X_resampled, y_resampled, figures_path)