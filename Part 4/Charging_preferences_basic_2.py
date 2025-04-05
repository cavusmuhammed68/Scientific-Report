import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define output path
output_path = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 4"
os.makedirs(output_path, exist_ok=True)

# File paths for the datasets
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastlee.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlandss.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name="Form Responses 1")
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name="Form Responses 1")

# Standardising column names by stripping spaces
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Renaming columns for consistency
rename_dict = {
    "First three digit postcode ": "First three digit postcode",
    "How do you think  charging facilities could be improved to better meet the needs of electric car users?":
        "How do you think charging facilities could be improved to better meet the needs of EV users?",
    "How do you rate the current distribution of  charging stations along major routes in West Midlands?":
        "How do you rate the current distribution of EV charging stations along major routes?",
    "Which one would you choose if your vehicle had 30% battery left? ":
        "Which one would you choose if your vehicle had 30% battery left?",
}

west_midlands_df.rename(columns=rename_dict, inplace=True)

# Relevant columns for the analysis
relevant_columns_standard = [
    "How would you prefer to receive information about the availability of charging stations?",
    "Which one would you choose if your vehicle had 30% battery left?",
    "What did you consider when choosing between the two chargers?",
    "How long are you willing to travel to get to a charging station?"
]

# Filter datasets and drop missing values
charging_preferences_newcastle = newcastle_df[relevant_columns_standard].dropna()
charging_preferences_west_midlands = west_midlands_df[relevant_columns_standard].dropna()

# Summarise Newcastle and West Midlands data
newcastle_summary = charging_preferences_newcastle.apply(lambda col: col.value_counts()).fillna(0)
west_midlands_summary = charging_preferences_west_midlands.apply(lambda col: col.value_counts()).fillna(0)

# Ensure consistent naming for plotting
newcastle_summary = newcastle_summary.rename(index={
    "Availability of charger on-route": "Availability of charger\n(on-route)",
    "Availability of charger at the end of next trip": "Availability of charger\n(at the end of the next trip)"
})
west_midlands_summary = west_midlands_summary.rename(index={
    "Availability of charger on-route": "Availability of charger\n(on-route)",
    "Availability of charger at the end of next trip": "Availability of charger\n(at the end of the next trip)"
})

# Define categories for visualization
categories_dict = {
    "How would you prefer to receive information about the availability of charging stations?":
        ["Mobile app", "In-car navigation system", "Online map"],
    "Which one would you choose if your vehicle had 30% battery left?":
        ["DC Charger", "AC Charger"],
    "What did you consider when choosing between the two chargers?": [
        "Cost",
        "Duration of charge",
        "Availability of charger\n(on-route)",
        "Availability of charger\n(at the end of the next trip)"
    ],
    "How long are you willing to travel to get to a charging station?":
        ["Less than 1km", "1km to 3km", "3km to 6km", "6km to 9km", "More than 9km"]
}

# Function to plot and save horizontal bar chart with values displayed
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
        newcastle_values = newcastle_summary.loc[categories, question].fillna(0)
        west_midlands_values = west_midlands_summary.loc[categories, question].fillna(0)

        # Y positions
        y_positions = np.arange(y_offset, y_offset + len(categories))
        y_ticks.extend(y_positions)
        y_labels.extend(categories)

        # Plot bars
        bars_ne = ax.barh(y_positions - bar_height / 2, newcastle_values, bar_height, label="North East" if y_offset == 0 else "", color=color_newcastle)
        bars_wm = ax.barh(y_positions + bar_height / 2, west_midlands_values, bar_height, label="West Midlands" if y_offset == 0 else "", color=color_west_midlands)

        # Annotate values above bars for specific categories
        for bar, value in zip(bars_ne, newcastle_values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{int(value)}", va='center', fontsize=15, color='black')

        for bar, value in zip(bars_wm, west_midlands_values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{int(value)}", va='center', fontsize=15, color='black')

        # Increment offset
        y_offset += len(categories) + 1  # Add space between question groups

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=16)
    ax.set_xlabel("Count", fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.set_title("Combined Preferences: North East vs West Midlands", fontsize=16)
    ax.legend(fontsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.6)  # Add grid for readability

    plt.tight_layout()
    save_path = os.path.join(output_path, "Horizontal_Bar_Chart.png")
    plt.savefig(save_path, dpi=600)
    plt.show()

# Generate visualization and save figure
plot_horizontal_bar_chart(output_path)
