import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

# Create a Hexbin plot to show clustering density
plt.figure(figsize=(8, 6))

# Use the cluster label values to color the hexbin plot
plt.hexbin(pca_components[:, 0], pca_components[:, 1], C=combined_df['Cluster'], cmap='viridis', gridsize=30, edgecolors='none')

plt.title('Hexbin Plot of Clustering (Density Plot)', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=14)
plt.ylabel('PCA Component 2', fontsize=14)

# Add colorbar to show the cluster labels
plt.colorbar(label='Cluster')

# Save the plot as a high-res image
save_path_hexbin = r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Hexbin_Cluster_Plot.png'
plt.savefig(save_path_hexbin, dpi=600)

# Show the plot
plt.show()


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the previous data and clusters are already computed and available

# Add PCA components to the dataframe for pairwise plotting
combined_df['PCA Component 1'] = pca_components[:, 0]
combined_df['PCA Component 2'] = pca_components[:, 1]

# Create a pairplot using seaborn
sns.pairplot(combined_df[['PCA Component 1', 'PCA Component 2', 'Cluster']], hue='Cluster', palette='viridis')
plt.suptitle('Pairwise Plot of Clusters', fontsize=16)
plt.savefig(r'C:\Users\cavus\Desktop\Dilum_Ekip_Paper\New Study\Pairwise_Cluster_Plot.png', dpi=600)
plt.show()

