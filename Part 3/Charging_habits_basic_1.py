import pandas as pd
import matplotlib.pyplot as plt
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
    "Histogram: Weekly Charging Frequency",
    "Histogram: Daily Travel Distance",
    "Histogram: Maximum Waiting Time Tolerance",
    "Histogram: Charging Time of Day",
    "Histogram: Weekly Charging Frequency (Detailed)",
    "Histogram: Average Charging Duration"
]

# Create histograms
fig, axes = plt.subplots(3, 2, figsize=(18, 15))

for i, question in enumerate(questions):
    row, col = divmod(i, 2)
    ax = axes[row, col]

    # Data preparation
    newcastle_data = newcastle_df[columns_to_check[question]].value_counts()
    west_midlands_data = west_midlands_df[columns_to_check[question]].value_counts()

    categories = sorted(list(set(newcastle_data.index).union(set(west_midlands_data.index))), key=str)
    newcastle_values = [newcastle_data.get(cat, 0) for cat in categories]
    west_midlands_values = [west_midlands_data.get(cat, 0) for cat in categories]

    # Plot histograms
    width = 0.35
    x = np.arange(len(categories))
    ax.bar(x - width/2, newcastle_values, width=width, label="Newcastle", color="blue")
    ax.bar(x + width/2, west_midlands_values, width=width, label="West Midlands", color="orange")

    # Add titles and labels
    ax.set_title(titles[i], size=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("Number of Responses", fontsize=16)
    ax.set_xlabel("Categories", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Histograms_All_Questions.png', dpi=600)
plt.show()
