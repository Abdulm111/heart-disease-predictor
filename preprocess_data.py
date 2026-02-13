import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the cleaned dataset
df = pd.read_csv('data/cardio_cleaned.csv')
print("Shape : ",df.shape)
print("Columns : ",df.columns)

# ---------- Feature Engineering: Create BMI ----------
df["bmi"] = df["weight"] / (df["height"]/100)**2
print("BMI Created")
print("BMI Range : ", df["bmi"].min(), '-', round(df["bmi"].max(),2))

# Split Features (X) and Target (y)
# we need to separate what the model will learn from (features) and what it will predict (target).
X = df.drop("cardio", axis=1) # Features
y = df["cardio"] # Target
print("Features Shape : ",X.shape)
print("Target Shape : ",y.shape)
print("Feature Columns : ",list(X.columns))

# Train-Test Split
#  split our data into Training (80%) and Testing (20%)!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train Shape : ",X_train.shape)
print('X_test Shape : ',X_test.shape)
print("y_train Shape : ",y_train.shape) 
print("y_test Shape : ",y_test.shape)

# Feature Scaling (StandardScaler)
scaler = StandardScaler()
# Fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)
# Only transform test data (NOT fit!)
X_test_scaled = scaler.transform(X_test)
print("Before scaling(first row of X_train) : ",X_train.iloc[0].values)
print("After scaling(first row of X_train) : ",X_train_scaled[0])
# Why fit_transform vs transform?
# X_train → scaler.fit_transform()                                │
# │            • fit = Learn the mean & std from training data        │
# │            • transform = Apply the scaling                       │
# │                                                                  │
# │  X_test  → scaler.transform()  (NO FIT!)                         │
# │            • Only apply the SAME scaling learned from training    │
# │                                                                  │
# │  WHY?                                                            │
# │  If we fit on test data too, we'd be "cheating" —                │
# │  the model would indirectly learn about test data!               │
# │  This is called DATA LEAKAGE ❌                                  │
# │                                                                  │
# │  Real world: You won't have future patient data to fit on!       │
# │  You scale new patients using the SAME scaler from training

# Save Preprocessed Data
data = {
    "X_train": X_train_scaled,
    "X_test": X_test_scaled,
    "y_train": y_train.values,
    "y_test": y_test.values,
    "scaler": scaler,
    "feature_names": list(X.columns)
}
with open('data/preprocessed_data.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Preprocessed data saved as 'data/preprocessed_data.pkl'")
print("Preprocessing complete!")
print(f"Training Samples : {X_train_scaled.shape[0]}")
print(f"Testing Samples : {X_test_scaled.shape[0]}")
print(f"Features : {X_train_scaled.shape[1]}")
print(f"Feature Names : {list(X.columns)}")