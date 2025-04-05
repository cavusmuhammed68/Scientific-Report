import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths for the datasets
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastlee.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlandss.xlsx'

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

# Summarize Newcastle and West Midlands data
newcastle_summary = charging_preferences_newcastle.apply(lambda col: col.value_counts()).fillna(0)
west_midlands_summary = charging_preferences_west_midlands.apply(lambda col: col.value_counts()).fillna(0)

# Combine all data for horizontal bar plots
categories_dict = {
    "Receive info about chargers": ["Mobile app", "In-car navigation system", "Online map"],
    "Battery_30%_Choice": ["DC Charger", "AC Charger"],
    "What did you consider for chargers?": [
        "Cost",
        "Duration of charge",
        "Availability of charger\n(on-route)",
        "Availability of charger\n(at the end of the next trip)"
    ],
    "Travel distance to charger": ["Less than 1km", "1km to 3km", "3km to 6km", "6km to 9km", "More than 9km"]
}

# Replace original labels for splitting
newcastle_summary = newcastle_summary.rename(index={
    "Availability of charger on-route": "Availability of charger\n(on-route)",
    "Availability of charger at the end of next trip": "Availability of charger\n(at the end of the next trip)"
})
west_midlands_summary = west_midlands_summary.rename(index={
    "Availability of charger on-route": "Availability of charger\n(on-route)",
    "Availability of charger at the end of next trip": "Availability of charger\n(at the end of the next trip)"
})

# Plot horizontal bar chart with grid and updated x-tick font size
def plot_horizontal_bar_chart(output_path):
    fig, ax = plt.subplots(figsize=(16, 12))
    
    bar_height = 0.35
    y_offset = 0
    y_ticks = []
    y_labels = []
    color_newcastle = "skyblue"
    color_west_midlands = "salmon"
    
    for question, categories in categories_dict.items():
        # Extract values for Newcastle and West Midlands
        newcastle_values = newcastle_summary.loc[categories, question]
        west_midlands_values = west_midlands_summary.loc[categories, question]
        
        # Y positions
        y_positions = np.arange(y_offset, y_offset + len(categories))
        y_ticks.extend(y_positions)
        y_labels.extend(categories)
        
        # Plot bars
        ax.barh(y_positions - bar_height / 2, newcastle_values, bar_height, label="North East" if y_offset == 0 else "", color=color_newcastle)
        ax.barh(y_positions + bar_height / 2, west_midlands_values, bar_height, label="West Midlands" if y_offset == 0 else "", color=color_west_midlands)
        
        # Increment offset
        y_offset += len(categories) + 1  # Add space between question groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.set_xlabel("Count", fontsize=16)
    ax.tick_params(axis='x', labelsize=16)  # Set x-tick font size
    ax.set_title("Combined Preferences: North East vs West Midlands", fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.6)  # Add grid for readability

    plt.tight_layout()
    plt.savefig(output_path + "\\Horizontal_Bar_Chart.png", dpi=600)
    plt.show()

# Define output path
output_path = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study"

# Generate visualisations
plot_horizontal_bar_chart(output_path)
