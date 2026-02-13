import pickle

# Load the preprocessed data
with open("data/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Save ONLY the scaler (tiny file!)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(data["scaler"], f)

print("Scaler saved to models/scaler.pkl ")