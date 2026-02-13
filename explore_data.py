import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv('data/cardio_train.csv', sep=';')
print("Dataset loaded successfully!")
print("Shape : ",df.shape)
#print the first five rows of the dataset
# print(df.head())
# Show all column names
print("Column names : ",df.columns)
# Show data types and non-null counts
# print(df.info())
# Shows statistical summary
# print(df.describe())
# Check the target distribution 
# print(df['cardio'].value_counts())

# Target Distribution Bar Chart
plt.figure(figsize=(8,5))
sns.countplot(x='cardio', data=df)
plt.title('Heart Disease Distribution')
plt.xlabel("0=No Disease, 1=Disease")
plt.ylabel('Count')
plt.savefig('images/target_distribution.png')
plt.show()
# Age Distribution Histogram
df["age_years"] = df["age"]/365.25
plt.figure(figsize=(8,5))
sns.histplot(df["age_years"], bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.savefig('images/age_distribution.png')
plt.show()

# Correlation Heatmap : It shows us which features are most related to heart disease.
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
plt.show()