import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')

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

# Define columns of interest
columns_of_interest = {
    "time_on_road": "How does charging station availability affect your time on the road?",
    "recharge_level": "At what charge level do you typically recharge your EV?",
    "planning_trips": "How do you include charging stations in planning trips over 50 km?",
    "rating_distribution": "EV charging station distribution rating",
    "parking_influence": "How has EV charging availability influenced your parking habits?",
    "parking_convenience": "How convenient is parking near charging stations versus regular spaces?"
}

# Extract relevant columns
west_midlands_filtered = west_midlands_df[list(columns_of_interest.values())]
newcastle_filtered = newcastle_df[list(columns_of_interest.values())]

# Standardize column names for comparison
west_midlands_filtered.columns = columns_of_interest.keys()
newcastle_filtered.columns = columns_of_interest.keys()

# Add region labels
west_midlands_filtered['region'] = 'West Midlands'
newcastle_filtered['region'] = 'Newcastle'

# Combine datasets
combined_data = pd.concat([west_midlands_filtered, newcastle_filtered], ignore_index=True)

# Custom x-tick labels for specific figures
custom_ticks = {
    "parking_convenience": [
        "slightly less\n(convenient)", 
        "much more\n(convenient)", 
        "slightly more\n(convenient)", 
        "much less\n(convenient)"
    ],
    "time_on_road": [
        "slightly \nincreases\n(travel time)",
        "slightly \ndecreases\n(travel time)",
        "significantly \nincreases\n(travel time)",
        "no impact \non\n(travel time)",
        "significantly \nreduces\n(travel time)"
    ],
    "planning_trips": [
        "convenient\n(stations)", 
        "backup\n(stations)", 
        "route around\n(stations)"
    ],
    "parking_influence": [
        "no impact \non my \nparking habits",
        "I can park \ncloser to \nmy destination",
        "I have to \npark further \nfrom \nmy destination"
    ]
}

# Set up the figure with 2 columns and 3 rows
fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=600)
axes = axes.flatten()  # Flatten axes for easier indexing

# Updated figure titles
figure_titles = [
    "How does charging station availability \naffect your time on the road?",
    "At what charge level do you \ntypically recharge your EV?",
    "How do you include charging stations \nin planning trips over 50 km?",
    "EV charging station distribution \nrating",
    "How has EV charging availability \ninfluenced your parking habits?",
    "How convenient is parking near \ncharging stations versus regular spaces?"
]

# Iterate through columns of interest and create plots
for i, (column, title) in enumerate(zip(columns_of_interest.keys(), figure_titles)):
    sns.violinplot(
        ax=axes[i],
        x='region', y=column, data=combined_data, cut=0, scale='count', inner=None, palette='muted'
    )
    sns.boxplot(
        ax=axes[i],
        x='region', y=column, data=combined_data, width=0.2, palette='dark', showcaps=True,
        boxprops={'zorder': 2}, whiskerprops={'zorder': 2}, showfliers=False
    )
    axes[i].set_title(title, fontsize=16)
    axes[i].set_ylabel(column.replace('_', ' ').capitalize(), fontsize=14)
    axes[i].set_xlabel("")  # Remove 'region' label from x-axis
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    
    # Apply custom y-tick labels if available
    if column in custom_ticks:
        axes[i].set_yticks(range(len(custom_ticks[column])))
        axes[i].set_yticklabels(custom_ticks[column], fontsize=12)

# Adjust layout to avoid overlapping
plt.tight_layout()

# Save the figure
save_path_additional = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Part 6\travel_pattern.png'
plt.savefig(save_path_additional, dpi=600)

# Show the plot
plt.show()


