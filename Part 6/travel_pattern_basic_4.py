# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 18:20:19 2025

@author: cavus
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define output path
output_path = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 6"
os.makedirs(output_path, exist_ok=True)

file_newcastle = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastlee.xlsx"
file_west_midlands = r"C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlandss.xlsx"

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name="Form Responses 1")
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name="Form Responses 1")

# Standardising column names by stripping spaces and unifying terminology
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Rename columns to standardise names across datasets
rename_dict = {
    "How do you rate the current distribution of EV charging stations along major routes in Newcastle?":
        "How do you rate the current distribution of EV charging stations along major routes?",
    "How do you rate the current distribution of  charging stations along major routes in West Midlands?":
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

# Apply renaming to datasets
newcastle_df.rename(columns=rename_dict, inplace=True)
west_midlands_df.rename(columns=rename_dict, inplace=True)

# Define columns of interest
columns_of_interest = {
    "time_on_road": "How do you think the availability of charging stations affects the time you spend on the road?",
    "recharge_level": "At what state of charge do you typically decide to recharge your EV?",
    "planning_trips": "When planning longer trips (over 50 km), how do you incorporate charging station locations into your route planning?",
    "rating_distribution": "How do you rate the current distribution of EV charging stations along major routes?",
    "parking_influence": "How has the availability of EV charging stations affected your parking habits?",
    "parking_convenience": "How convenient do you find parking near available charging stations compared to regular parking spots?"
}

# Extract relevant columns, dropping rows with missing values
west_midlands_filtered = west_midlands_df[list(columns_of_interest.values())].dropna()
newcastle_filtered = newcastle_df[list(columns_of_interest.values())].dropna()

# Standardise column names for merging
west_midlands_filtered.columns = columns_of_interest.keys()
newcastle_filtered.columns = columns_of_interest.keys()

# Add region labels
west_midlands_filtered["region"] = "West Midlands"
newcastle_filtered["region"] = "North East"

# Combine datasets
combined_data = pd.concat([west_midlands_filtered, newcastle_filtered], ignore_index=True)

# Define custom y-tick labels
custom_ticks = {
    "parking_convenience": [
        "Slightly less\nconvenient", 
        "Much more\nconvenient", 
        "Slightly more\nconvenient", 
        "Much less\nconvenient"
    ],
    "time_on_road": [
        "Slightly \nincreases\n(travel time)",
        "Slightly \ndecreases\n(travel time)",
        "Significantly \nincreases\n(travel time)",
        "No impact \non\n(travel time)",
        "Significantly \nreduces\n(travel time)"
    ],
    "planning_trips": [
        "Convenient\nstations", 
        "Backup\nstations", 
        "Route around\nstations"
    ],
    "parking_influence": [
        "No impact \non my \nparking habits",
        "I can park \ncloser to \nmy destination",
        "I have to \npark further \nfrom \nmy destination"
    ]
}

# Set up the figure with 2 columns and 3 rows
fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=600)
axes = axes.flatten()  # Flatten axes for easier iteration

# Iterate through columns of interest and create violin-box plots
for i, column in enumerate(columns_of_interest.keys()):
    sns.violinplot(
        ax=axes[i],
        x="region", y=column, data=combined_data, cut=0, scale="count", inner=None, palette="muted"
    )
    sns.boxplot(
        ax=axes[i],
        x="region", y=column, data=combined_data, width=0.2, palette="dark", showcaps=True,
        boxprops={"zorder": 2}, whiskerprops={"zorder": 2}, showfliers=False
    )
    axes[i].set_title(column.replace("_", " ").capitalize(), fontsize=16)
    axes[i].set_ylabel(column.replace("_", " ").capitalize(), fontsize=14)
    axes[i].set_xlabel("")  # Remove 'region' label from x-axis
    axes[i].tick_params(axis="x", labelsize=12)
    axes[i].tick_params(axis="y", labelsize=12)
    
    # Apply custom y-tick labels if available
    if column in custom_ticks:
        tick_positions = range(len(custom_ticks[column]))
        axes[i].set_yticks(tick_positions)
        axes[i].set_yticklabels(custom_ticks[column], fontsize=12)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the figure
save_path = os.path.join(output_path, "travel_pattern.png")
plt.savefig(save_path, dpi=600)

# Display the plot
plt.show()
