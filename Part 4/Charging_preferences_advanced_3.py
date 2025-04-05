import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Purpose: To explore how users select charging stations and their preferences for accessibility.

# File paths for the datasets
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')

# Standardising column names by stripping leading/trailing spaces
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Correcting column names with new column names
newcastle_df = newcastle_df.rename(columns={"Receive info about chargers": "Receive info about chargers",
                                             "Battery 30%_Choice": "Battery_30%_Choice",
                                             "What did you consider for chargers?": "What did you consider for chargers?",
                                             "Travel distance to charger": "Travel distance to charger"})
west_midlands_df = west_midlands_df.rename(columns={"Receive info about chargers": "Receive info about chargers",
                                                    "Battery 30%_Choice": "Battery_30%_Choice",
                                                    "What did you consider for chargers?": "What did you consider for chargers?",
                                                    "Travel distance to charger": "Travel distance to charger"})

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

# Summarize Newcastle and West Midlands data
newcastle_summary = charging_preferences_newcastle.apply(lambda col: col.value_counts()).fillna(0)
west_midlands_summary = charging_preferences_west_midlands.apply(lambda col: col.value_counts()).fillna(0)

# Categories dictionary
categories_dict = {
    "Receive info about chargers": ["Mobile app", "In-car navigation system", "Online map"],
    "Battery_30%_Choice": ["DC Charger", "AC Charger"],
    "What did you consider for chargers?": [
        "Cost",
        "Duration of charge",
        "Availability of charger on-route",
        "Availability of charger at the end of next trip"
    ],
    "Travel distance to charger": ["Less than 1km", "1km to 3km", "3km to 6km", "6km to 9km", "More than 9km"]
}

# Plot horizontal bar chart comparisons
def plot_horizontal_bar_chart_inline(newcastle_summary, west_midlands_summary, categories_dict):
    fig, ax = plt.subplots(figsize=(16, 12))

    bar_height = 0.35
    y_offset = 0
    y_ticks = []
    y_labels = []
    color_newcastle = "skyblue"
    color_west_midlands = "salmon"

    for question, categories in categories_dict.items():
        # Extract values for Newcastle and West Midlands
        newcastle_values = newcastle_summary.loc[categories, question].fillna(0)
        west_midlands_values = west_midlands_summary.loc[categories, question].fillna(0)

        # Y positions
        y_positions = np.arange(y_offset, y_offset + len(categories))
        y_ticks.extend(y_positions)
        y_labels.extend(categories)

        # Plot bars
        ax.barh(y_positions - bar_height / 2, newcastle_values, bar_height, label="Newcastle" if y_offset == 0 else "", color=color_newcastle)
        ax.barh(y_positions + bar_height / 2, west_midlands_values, bar_height, label="West Midlands" if y_offset == 0 else "", color=color_west_midlands)

        # Increment offset
        y_offset += len(categories) + 1  # Add space between question groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel("Count", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)  # Set x-tick font size
    ax.set_title("Comparison of Preferences: Newcastle vs West Midlands", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)  # Add grid for readability

    plt.tight_layout()
    plt.show()

# Generate the bar chart
plot_horizontal_bar_chart_inline(newcastle_summary, west_midlands_summary, categories_dict)

# ML Application: Decision-tree models to analyse user decision-making
def run_decision_tree(data, region_name, output_path):
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
    clf = DecisionTreeClassifier(random_state=42, max_depth=3)  # Limiting depth for clarity
    clf.fit(X_train, y_train)

    # Convert class names to list
    class_names = list(label_encoders["Battery_30%_Choice"].classes_)

    # Plot Decision Tree
    plt.figure(figsize=(20, 16))  # Larger figure for better readability
    plot_tree(
        clf, 
        feature_names=feature_names,  # Fixed: Converted to list
        class_names=class_names,  # Converted to list
        filled=True,
        rounded=True,
        fontsize=12
    )
    plt.title(f"Decision Tree for {region_name} Preferences", fontsize=18)
    plt.savefig(f"{output_path}/Decision_Tree_{region_name}.png", dpi=300)
    plt.show()

    return clf


# Output path for saving decision tree figures
output_path = r"C:\\Users\\cavus\\Desktop\\Dilum_Ekip_Paper\\New Study"

# Train and visualize decision trees for each region
decision_tree_newcastle = run_decision_tree(charging_preferences_newcastle, "Newcastle", output_path)
decision_tree_west_midlands = run_decision_tree(charging_preferences_west_midlands, "West Midlands", output_path)
