import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('diabetic_data.csv')

missing_count = df.isnull().sum()
#print(missing_count)

df['race'] = df['race'].replace('?','Unknown')

df = df.replace('?','Unknown')
#print(df.head(50))

# Removed because of lot of missing values so no use of these columns
df.drop(columns=['max_glu_serum','A1Cresult'], inplace = True) 

final_missing_count = df.isnull().sum()
#print(final_missing_count)

#Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Create the correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

# Save the heatmap to a file
plt.savefig('correlation_heatmap.png', dpi=300)

# Close the plot
plt.close()