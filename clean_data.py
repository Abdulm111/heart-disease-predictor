import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv('data/cardio_train.csv', sep=';')
print("Original Shape : ",df.shape)
# Drop the id column as it is not useful for modeling
df =df.drop('id',axis=1)
print("Shape after dropping 'id' column : ",df.shape)
# Convert Age from Days → Years
df["age"] = (df["age"]/365.25).astype(int)
print('Age converted in to years!')
print("Age range : ",df["age"].min(),"-",df["age"].max())

# Remove Outliers — Blood Pressure
# Check blood pressure BEFORE cleaning
print("ap_hi range : ", df["ap_hi"].min(), "-", df["ap_hi"].max())
print("ap_lo range : ", df["ap_lo"].min(), "-", df["ap_lo"].max())
# Remove blood pressure outliers
print("Shape before removing blood pressure outliers : ",df.shape)
df = df[(df["ap_hi"]>=80) & (df["ap_hi"]<=200)]
df = df[(df["ap_lo"]>=50) & (df["ap_lo"]<=140)]
df = df[df["ap_hi"] > df["ap_lo"]]
print("Shape after removing blood pressure outliers : ",df.shape)

# Check height & weight BEFORE cleaning
print("Height Range : ", df["height"].min(),"-", df["height"].max())
print("Weight Range : ", df["weight"].min(),"-", df["weight"].max())
# Remove height outliers (keep reasonable adult heights)
print("Shape before cleaning height/weight : ",df.shape)
df = df[(df['height']>=120) & (df["height"]<=210)]
df = df[(df['weight']>=30) & (df["weight"]<=200)]
print("Shape after cleaning height/weight : ",df.shape)

# ---------- Final Verification ----------
print("Final Shape : ",df.shape)
print("Age Range : ",df["age"].min(),"-",df["age"].max())
print("ap_hi range : ", df["ap_hi"].min(), "-", df["ap_hi"].max())
print("ap_lo range : ", df["ap_lo"].min(), "-", df["ap_lo"].max())
print("Height Range : ", df["height"].min(),"-", df["height"].max())
print("Weight Range : ", df["weight"].min(),"-", df["weight"].max())
print("Target Distibution : ")
print(df["cardio"].value_counts())
# Save the cleaned dataset
df.to_csv('data/cardio_cleaned.csv', index=False)
print("Cleaned dataset saved as 'data/cardio_cleaned.csv'")