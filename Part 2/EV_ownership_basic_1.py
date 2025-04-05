import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# Load the datasets from the provided paths
west_midlands_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_West_Midlands.xlsx')
newcastle_data = pd.ExcelFile(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Survey_Newcastle.xlsx')

# Parse the sheets
west_midlands_df = west_midlands_data.parse('Form Responses 1')
newcastle_df = newcastle_data.parse('Form Responses 1')

# Clean the data for relevant columns
west_midlands_df.columns = west_midlands_df.columns.str.strip()  # Clean column names
newcastle_df.columns = newcastle_df.columns.str.strip()  # Clean column names

# Filter the relevant columns and drop missing values
west_midlands_cleaned = west_midlands_df[['Motivation behind buying EV?', 'What type of EV do you own?', 
                                          'What is the approximate all-electric range of your EV?', 
                                          'How long have you been driving an EV?']].dropna()

newcastle_cleaned = newcastle_df[['Motivation behind buying EV?', 'What type of EV do you own?', 
                                  'What is the approximate all-electric range of your EV?', 
                                  'How long have you been driving an EV?']].dropna()

# Add a column to identify the region
west_midlands_cleaned['Region'] = 'West Midlands'
newcastle_cleaned['Region'] = 'Newcastle'

# Combine both datasets
combined_df = pd.concat([west_midlands_cleaned, newcastle_cleaned])

# Separate the Region column to avoid scaling issues
region_column = combined_df['Region']
combined_df_encoded = combined_df.drop(columns=['Region'])

# One-hot encoding the categorical variables
combined_df_encoded = pd.get_dummies(combined_df_encoded, columns=['Motivation behind buying EV?', 'What type of EV do you own?',
                                                           'What is the approximate all-electric range of your EV?', 
                                                           'How long have you been driving an EV?'])

# Standardize the data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_df_encoded)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Assume 3 clusters
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original data
combined_df['Cluster'] = clusters
combined_df['Region'] = region_column  # Add Region back for visualization

# Use PCA for dimensionality reduction to visualize high-dimensional data
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Plotting the clusters for both regions in separate subplots within the same figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Filter data for Newcastle and West Midlands
newcastle_df_for_plot = combined_df[combined_df['Region'] == 'Newcastle']
west_midlands_df_for_plot = combined_df[combined_df['Region'] == 'West Midlands']

# PCA Components for Newcastle
newcastle_pca = pca_components[combined_df['Region'] == 'Newcastle']

# PCA Components for West Midlands
west_midlands_pca = pca_components[combined_df['Region'] == 'West Midlands']

# Plot the Newcastle clustering
scatter1 = axes[0].scatter(newcastle_pca[:, 0], newcastle_pca[:, 1], c=newcastle_df_for_plot['Cluster'], cmap='viridis', edgecolors='k', alpha=0.7)
axes[0].set_title('Clustering of Newcastle EV Users', fontsize=16)
axes[0].set_xlabel('PCA Component 1', fontsize=16)
axes[0].set_ylabel('PCA Component 2', fontsize=16)
axes[0].tick_params(axis='x', rotation=45, labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)

# Plot the West Midlands clustering
scatter2 = axes[1].scatter(west_midlands_pca[:, 0], west_midlands_pca[:, 1], c=west_midlands_df_for_plot['Cluster'], cmap='viridis', edgecolors='k', alpha=0.7)
axes[1].set_title('Clustering of West Midlands EV Users', fontsize=16)
axes[1].set_xlabel('PCA Component 1', fontsize=16)
axes[1].set_ylabel('PCA Component 2', fontsize=16)
axes[1].tick_params(axis='x', rotation=45, labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)

# Add colorbar (cluster reference) to the right of the figures (linked with the clusters)
fig.colorbar(scatter1, ax=axes, orientation='vertical', label='Cluster')

# Adjust layout to ensure everything fits properly
#plt.subplots_adjust(right=0.85)  # Move colorbar to the right

# Save the clustering plot as a high-res image
save_path = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Cluster_comparison_separate_regions_with_colorbar.png'
plt.savefig(save_path, dpi=600)

# Show the clustering plot
plt.show()

# Now, let's add the additional figures (Motivations, EV Types, Range, and Driving Time)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Motivation behind buying an EV
motivation_counts_newcastle = newcastle_cleaned['Motivation behind buying EV?'].value_counts()
motivation_counts_wm = west_midlands_cleaned['Motivation behind buying EV?'].value_counts()

motivation_counts_newcastle.plot(kind='bar', ax=axes[0, 0], color='lightblue', width=0.4, position=1, label='Newcastle')
motivation_counts_wm.plot(kind='bar', ax=axes[0, 0], color='lightgreen', width=0.4, position=0, label='West Midlands')
axes[0, 0].set_title('Motivation Behind Buying an EV', fontsize=16)
axes[0, 0].set_xlabel('Motivation', fontsize=16)
axes[0, 0].set_ylabel('Count', fontsize=16)
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=14)
axes[0, 0].tick_params(axis='y', labelsize=14)
axes[0, 0].grid(True)

# Type of EV owned
ev_type_counts_newcastle = newcastle_cleaned['What type of EV do you own?'].value_counts()
ev_type_counts_wm = west_midlands_cleaned['What type of EV do you own?'].value_counts()

ev_type_counts_newcastle.plot(kind='bar', ax=axes[0, 1], color='lightblue', width=0.4, position=1, label='Newcastle')
ev_type_counts_wm.plot(kind='bar', ax=axes[0, 1], color='lightgreen', width=0.4, position=0, label='West Midlands')
axes[0, 1].set_title('Type of EV Owned', fontsize=16)
axes[0, 1].set_xlabel('Type of EV', fontsize=16)
axes[0, 1].set_ylabel('Count', fontsize=16)
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=14)
axes[0, 1].tick_params(axis='y', labelsize=14)
axes[0, 1].grid(True)

# Approximate all-electric range of the EV
range_counts_newcastle = newcastle_cleaned['What is the approximate all-electric range of your EV?'].value_counts()
range_counts_wm = west_midlands_cleaned['What is the approximate all-electric range of your EV?'].value_counts()

range_counts_newcastle.plot(kind='bar', ax=axes[1, 0], color='lightblue', width=0.4, position=1, label='Newcastle')
range_counts_wm.plot(kind='bar', ax=axes[1, 0], color='lightgreen', width=0.4, position=0, label='West Midlands')
axes[1, 0].set_title('Approximate All-Electric Range of the EV', fontsize=16)
axes[1, 0].set_xlabel('Range', fontsize=16)
axes[1, 0].set_ylabel('Count', fontsize=16)
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=14)
axes[1, 0].tick_params(axis='y', labelsize=14)
axes[1, 0].grid(True)

# How long respondents have been driving an EV
driving_time_counts_newcastle = newcastle_cleaned['How long have you been driving an EV?'].value_counts()
driving_time_counts_wm = west_midlands_cleaned['How long have you been driving an EV?'].value_counts()

driving_time_counts_newcastle.plot(kind='bar', ax=axes[1, 1], color='lightblue', width=0.4, position=1, label='Newcastle')
driving_time_counts_wm.plot(kind='bar', ax=axes[1, 1], color='lightgreen', width=0.4, position=0, label='West Midlands')
axes[1, 1].set_title('How Long Have You Been Driving an EV?', fontsize=16)
axes[1, 1].set_xlabel('Time Period', fontsize=16)
axes[1, 1].set_ylabel('Count', fontsize=16)
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=14)
axes[1, 1].tick_params(axis='y', labelsize=14)
axes[1, 1].grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the additional figures as a high-res image
save_path_additional = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Additional_Figures_with_Grid.png'
plt.savefig(save_path_additional, dpi=600)

# Show the additional comparison figure
plt.show()
