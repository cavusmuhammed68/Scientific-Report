import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# Use a different matplotlib style
style.use('ggplot')

file_newcastle = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx'
file_west_midlands = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx'

# Load Excel files
newcastle_df = pd.read_excel(file_newcastle, sheet_name='Form Responses 1')
west_midlands_df = pd.read_excel(file_west_midlands, sheet_name='Form Responses 1')

# Strip column names to remove trailing/leading spaces
newcastle_df.columns = newcastle_df.columns.str.strip()
west_midlands_df.columns = west_midlands_df.columns.str.strip()

# Create combined figure with all subplots
fig, axes = plt.subplots(3, 2, figsize=(18, 18))

# Define plot-specific adjustments for better clarity
bar_width = 0.4
alpha_value = 0.8

# Satisfaction Visualization
satisfaction_labels = ["1: Very Dissatisfied", "2: Dissatisfied", "3: Neutral", "4: Satisfied", "5: Very Satisfied"]
newcastle_satisfaction = newcastle_df['User satisfaction with charging station availability'].value_counts().sort_index()
west_midlands_satisfaction = west_midlands_df['User satisfaction with charging station availability'].value_counts().sort_index()

newcastle_satisfaction = newcastle_satisfaction.reindex(range(1, 6), fill_value=0)
west_midlands_satisfaction = west_midlands_satisfaction.reindex(range(1, 6), fill_value=0)

x_indexes = range(1, 6)
axes[0, 0].bar(x_indexes, newcastle_satisfaction, alpha=alpha_value, label="Newcastle", color='steelblue', width=bar_width)
axes[0, 0].bar([x + bar_width for x in x_indexes], west_midlands_satisfaction, alpha=alpha_value, label="West Midlands", color='sandybrown', width=bar_width)
axes[0, 0].set_xticks([x + bar_width / 2 for x in x_indexes])
axes[0, 0].set_xticklabels(satisfaction_labels, fontsize=12, rotation=45, ha='right')
axes[0, 0].set_xlabel("Satisfaction Level", fontsize=14)
axes[0, 0].set_ylabel("Number of Responses", fontsize=14)
axes[0, 0].set_title("User Satisfaction with Charging Station Availability", fontsize=16)
axes[0, 0].legend()

# Experienced Difficulties Visualization
labels = ['Yes', 'No']
newcastle_difficulties = newcastle_df['Experienced difficulties finding charging station'].value_counts()
west_midlands_difficulties = west_midlands_df['Experienced difficulties finding charging station'].value_counts()

axes[0, 1].bar(range(len(labels)), [newcastle_difficulties.get(label, 0) for label in labels], alpha=alpha_value, label="Newcastle", color='steelblue', width=bar_width)
axes[0, 1].bar([i + bar_width for i in range(len(labels))], [west_midlands_difficulties.get(label, 0) for label in labels], alpha=alpha_value, label="West Midlands", color='sandybrown', width=bar_width)
axes[0, 1].set_xticks([x + bar_width / 2 for x in range(len(labels))])
axes[0, 1].set_xticklabels(labels, fontsize=12)
axes[0, 1].set_xlabel("Response", fontsize=14)
axes[0, 1].set_ylabel("Number of Responses", fontsize=14)
axes[0, 1].set_title("Experienced Difficulties Finding Charging Station", fontsize=16)
axes[0, 1].legend()

# Importance Visualization
importance_labels = ["1: Not Important", "2", "3", "4", "5: Very Important"]
newcastle_importance = newcastle_df['How important is charging station when planning travel?'].value_counts().sort_index()
west_midlands_importance = west_midlands_df['How important is charging station when planning travel?'].value_counts().sort_index()

newcastle_importance = newcastle_importance.reindex(range(1, 6), fill_value=0)
west_midlands_importance = west_midlands_importance.reindex(range(1, 6), fill_value=0)

axes[1, 0].bar(x_indexes, newcastle_importance, alpha=alpha_value, label="Newcastle", color='steelblue', width=bar_width)
axes[1, 0].bar([x + bar_width for x in x_indexes], west_midlands_importance, alpha=alpha_value, label="West Midlands", color='sandybrown', width=bar_width)
axes[1, 0].set_xticks([x + bar_width / 2 for x in x_indexes])
axes[1, 0].set_xticklabels(importance_labels, fontsize=12, rotation=45, ha='right')
axes[1, 0].set_xlabel("Importance Level", fontsize=14)
axes[1, 0].set_ylabel("Number of Responses", fontsize=14)
axes[1, 0].set_title("Importance of Charging Station When Planning Travel", fontsize=16)
axes[1, 0].legend()

# Factors Influencing Choice
categories_factors = ["Proximity to my destination", "Availability of fast charging", "Cost of charging", "User-friendly interface", 
                      "Safety of the location", "Amenities available nearby", "Environmental considerations"]
newcastle_factors = pd.Series([123, 122, 109, 46, 61, 60, 21], index=categories_factors)
west_midlands_factors = pd.Series([116, 98, 102, 46, 56, 65, 18], index=categories_factors)

x_indexes_factors = range(len(categories_factors))
axes[1, 1].bar(x_indexes_factors, newcastle_factors, alpha=alpha_value, label="Newcastle", color='steelblue', width=bar_width)
axes[1, 1].bar([x + bar_width for x in x_indexes_factors], west_midlands_factors, alpha=alpha_value, label="West Midlands", color='sandybrown', width=bar_width)
axes[1, 1].set_xticks([x + bar_width / 2 for x in x_indexes_factors])
axes[1, 1].set_xticklabels(categories_factors, fontsize=12, rotation=45, ha='right')
axes[1, 1].set_xlabel("Factors", fontsize=14)
axes[1, 1].set_ylabel("Frequency", fontsize=14)
axes[1, 1].set_title("Factors Influencing Choice of Charging Stations", fontsize=16)
axes[1, 1].legend()

# Suggestions Visualization
categories_suggestions = ["Allow for longer charging \n(sessions)", 
                          "Enhance user experience through \n(better signage and information)", 
                          "Improve reliability and maintenance \n(of existing stations)", 
                          "Increase the number \n(of charging stations)"]
newcastle_suggestions = pd.Series([7, 8, 25, 109], index=categories_suggestions)
west_midlands_suggestions = pd.Series([6, 11, 31, 97], index=categories_suggestions)

x_indexes_suggestions = range(len(categories_suggestions))
axes[2, 0].bar(x_indexes_suggestions, newcastle_suggestions, alpha=alpha_value, label="Newcastle", color='steelblue', width=bar_width)
axes[2, 0].bar([x + bar_width for x in x_indexes_suggestions], west_midlands_suggestions, alpha=alpha_value, label="West Midlands", color='sandybrown', width=bar_width)
axes[2, 0].set_xticks([x + bar_width / 2 for x in x_indexes_suggestions])
axes[2, 0].set_xticklabels(categories_suggestions, fontsize=12, rotation=45, ha='right')
axes[2, 0].set_xlabel("Suggestions", fontsize=14)
axes[2, 0].set_ylabel("Frequency", fontsize=14)
axes[2, 0].set_title("Suggestions for Improving EV Charging Facilities", fontsize=16)
axes[2, 0].legend()

# Hide the empty subplot
fig.delaxes(axes[2, 1])

plt.tight_layout()
plt.savefig("combined_figures_updated.png", dpi=600)
plt.close()

# Question 1: User Satisfaction (Combined)
combined_satisfaction = newcastle_satisfaction + west_midlands_satisfaction
plt.figure(figsize=(8, 8))
plt.pie(combined_satisfaction, labels=satisfaction_labels, autopct='%1.1f%%', startangle=140)
plt.title("Combined: Satisfaction with Charging Stations")
plt.savefig("pie_chart_satisfaction.png", dpi=600)
plt.close()

# Question 2: Experienced Difficulties Finding Charging Stations (Combined)
combined_difficulties = [newcastle_difficulties.get(label, 0) + west_midlands_difficulties.get(label, 0) for label in labels]
plt.figure(figsize=(8, 8))
plt.pie(combined_difficulties, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Combined: Difficulties Finding Charging Stations")
plt.savefig("pie_chart_difficulties.png", dpi=600)
plt.close()

# Question 3: Importance of Charging Stations (Combined)
combined_importance = newcastle_importance + west_midlands_importance
plt.figure(figsize=(8, 8))
plt.pie(combined_importance, labels=importance_labels, autopct='%1.1f%%', startangle=140)
plt.title("Combined: Importance of Charging Stations")
plt.savefig("pie_chart_importance.png", dpi=600)
plt.close()

# Updated Factors Influencing Choice labels
categories_factors = [
    "Amenities available \nnearby",
    "Environmental \nconsideration",
    "Proximity to \nmy destination",
    "Availability of \nfast charging",
    "User-friendly \ninterface",
    "Safety of the \nlocation",
    "Cost of \ncharging"
]

newcastle_factors = pd.Series([60, 21, 123, 122, 46, 61, 109], index=categories_factors)
west_midlands_factors = pd.Series([65, 18, 116, 98, 46, 56, 102], index=categories_factors)

# Question 4: Factors Influencing Choice (Combined)
combined_factors = newcastle_factors + west_midlands_factors

# Save as a standalone pie chart
plt.figure(figsize=(8, 8))
plt.pie(combined_factors, labels=categories_factors, autopct='%1.1f%%', startangle=140)
plt.title("Combined: Factors Influencing Choice of Charging Stations")
plt.savefig("pie_chart_factors_updated.png", dpi=600)
plt.close()


# Updated Suggestions labels
categories_suggestions = [
    "Allow for longer charging \nsessions",
    "Enhance user \nexperience through \nbetter signage \nand information",
    "Improve reliability \nand maintenance \nof existing stations",
    "Increase the \nnumber of \ncharging stations"
]

newcastle_suggestions = pd.Series([7, 8, 25, 109], index=categories_suggestions)
west_midlands_suggestions = pd.Series([6, 11, 31, 97], index=categories_suggestions)

# Combine Newcastle and West Midlands data
combined_suggestions = newcastle_suggestions + west_midlands_suggestions

# Save as a standalone pie chart
plt.figure(figsize=(8, 8))
plt.pie(combined_suggestions, labels=categories_suggestions, autopct='%1.1f%%', startangle=140)
plt.title("Combined: Suggestions for Improving EV Charging Facilities")
plt.savefig("pie_chart_suggestions_updated.png", dpi=600)
plt.close()




