import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets from the provided paths
west_midlands_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx')
newcastle_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx')

# Parse the sheets
west_midlands_df = west_midlands_data.parse('Form Responses 1')
newcastle_df = newcastle_data.parse('Form Responses 1')

# Column mapping to requested questions
columns_to_check = {
    "frequency_public_charging": "How often in one week do you charge your vehicle?",
    "daily_travel_distance": "What is your daily average travel distance?",
    "waiting_time_tolerance": "Given there’s a charging point nearby but it’s occupied by another vehicle, what is the maximum timing you’re willing to wait for your turn?",
    "charging_time_of_day": "At what time of day do you usually start charging your EV?",
    "charging_frequency_weekly": "How often in one week do you charge your vehicle?",
    "charging_duration": "What is the average duration of your charge?"
}

# Questions and titles for analysis
questions = [
    "frequency_public_charging",
    "daily_travel_distance",
    "waiting_time_tolerance",
    "charging_time_of_day",
    "charging_frequency_weekly",
    "charging_duration"
]

titles = [
    "Weekly Charging Frequency",
    "Daily Travel Distance",
    "Maximum Waiting Time Tolerance",
    "Charging Time of Day",
    "Weekly Charging Frequency (Detailed)",
    "Average Charging Duration"
]

# Create heatmaps
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

for i, question in enumerate(questions):
    row, col = divmod(i, 2)
    ax = axes[row, col]

    # Data preparation
    newcastle_data = newcastle_df[columns_to_check[question]].value_counts()
    west_midlands_data = west_midlands_df[columns_to_check[question]].value_counts()

    categories = sorted(list(set(newcastle_data.index).union(set(west_midlands_data.index))), key=str)
    heatmap_data = pd.DataFrame({
        "Newcastle": [newcastle_data.get(cat, 0) for cat in categories],
        "West Midlands": [west_midlands_data.get(cat, 0) for cat in categories]
    }, index=categories)

    # Plot heatmap
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm", ax=ax, cbar=True, linewidths=0.5,
                annot_kws={"fontsize": 16})


    # Add titles and labels
    ax.set_title(titles[i], size=15)
    ax.set_xlabel("", fontsize=16)
    #ax.set_ylabel(columns_to_check[question], fontsize=16)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Heatmaps_All_Questions.png', dpi=600)
plt.show()
