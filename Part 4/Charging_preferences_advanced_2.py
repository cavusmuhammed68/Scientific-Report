import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# File paths for the datasets
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')

# Standardising column names by stripping leading/trailing spaces
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Renaming specific columns for consistency
newcastle_df = newcastle_df.rename(columns={
    "Battery 30%_Choice": "Battery_30%_Choice"
})
west_midlands_df = west_midlands_df.rename(columns={
    "Battery 30%_Choice": "Battery_30%_Choice"
})

# Relevant columns for the analysis
relevant_columns_standard = [
    "Receive info about chargers",
    "Battery_30%_Choice",
    "What did you consider for chargers?",
    "Travel distance to charger"
]

# Filter datasets
charging_preferences_newcastle = newcastle_df[relevant_columns_standard].dropna()
charging_preferences_west_midlands = west_midlands_df[relevant_columns_standard].dropna()

# Standardize the text values to avoid mismatches
for column in relevant_columns_standard:
    charging_preferences_newcastle[column] = charging_preferences_newcastle[column].str.strip()
    charging_preferences_west_midlands[column] = charging_preferences_west_midlands[column].str.strip()

# Function to run and plot decision tree
def run_and_plot_decision_tree(data, region_name, output_path):
    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    # Features and target
    X = data.drop(columns="Battery_30%_Choice")
    y = data["Battery_30%_Choice"]

    # Convert feature names to list
    feature_names = list(X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Convert class names to list
    class_names = list(label_encoders["Battery_30%_Choice"].classes_)

    # Plot Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=feature_names,  # Fixed: Converted to list
        class_names=class_names,  # Converted to list
        filled=True
    )
    plt.title(f"Decision Tree for {region_name} Preferences")
    plt.savefig(output_path + f"\\Decision_Tree_{region_name}.png", dpi=600)
    plt.show()

    return clf


# Define output path
output_path = r"C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\New Study"

# Run decision tree for Newcastle
decision_tree_newcastle = run_and_plot_decision_tree(
    charging_preferences_newcastle.copy(), "Newcastle", output_path
)

# Run decision tree for West Midlands
decision_tree_west_midlands = run_and_plot_decision_tree(
    charging_preferences_west_midlands.copy(), "West_Midlands", output_path
)