import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the preprocessed data
with open('data/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
print("Data loaded successfully.")  
print(f"Training : {X_train.shape[0] } samples ")
print(f"Testing : {X_test.shape[0] } samples ")

# Train Model 1 — KNN (K-Nearest Neighbors) 
# Craete the model 
knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
# Train(fit) the model 
knn.fit(X_train, y_train)
# Predict on test data 
knn_pred = knn.predict(X_test)
# Check the performance of the model
print("KNN Classification Report:")
print(classification_report(y_test, knn_pred))
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))
print("KNN Accuracy Score:", accuracy_score(y_test, knn_pred))

# Train Model 2 — Naive Bayes
# Create the model
nb = GaussianNB()
# Train(fit) the model
nb.fit(X_train, y_train)
# Predict on test data
nb_pred = nb.predict(X_test)
# Check the performance of the model
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_pred))
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_pred))
print("Naive Bayes Accuracy Score:", accuracy_score(y_test, nb_pred))

# Train Model 3 — Logistic Regression
# Create the model
lr = LogisticRegression(max_iter=1000)
# Train(fit) the model
lr.fit(X_train, y_train)
# Predict on test data
lr_pred = lr.predict(X_test)
# Check the performance of the model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print("Logistic Regression Accuracy Score:", accuracy_score(y_test, lr_pred))

# Train Model 4 — Support Vector Machine (SVM)
# Create the model
svm = SVC(kernel='rbf')
# Train(fit) the model
svm.fit(X_train, y_train)
# Predict on test data
svm_pred = svm.predict(X_test)
# Check the performance of the model
print("SVM Classification Report:")
print(classification_report(y_test, svm_pred))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))
print("SVM Accuracy Score:", accuracy_score(y_test, svm_pred))

# Train Model 5 — Random Forest
# Create the model
rf = RandomForestClassifier(n_estimators=50, max_depth=10,random_state=42)
# Train(fit) the model
rf.fit(X_train, y_train)
# Predict on test data
rf_pred = rf.predict(X_test)
# Check the performance of the model
print("Random Forest Classification Report:")
print(classification_report(y_test,rf_pred))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Random Forest Accuracy Score:", accuracy_score(y_test, rf_pred))

# Train Model 6 — XGBoost
# Create the model
xgb = XGBClassifier(n_estimators = 100, random_state=42, eval_metric = "logloss")
# Train(fit) the model
xgb.fit(X_train, y_train)
# Predict on test data
xgb_pred = xgb.predict(X_test)
# Check the performance of the model
print("XGBoost Classification Report:")
print(classification_report(y_test,xgb_pred))
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))
print("XGBoost Accuracy Score:", accuracy_score(y_test, xgb_pred))

# Save the Best Model + Comparison Summary
models = {
    "knn" : knn,
    "naive_bayes" : nb,
    "logistic_regression" : lr,
    "svm" : svm,
    "random_forest" : rf,
    "xgboost" : xgb
}
for name , model in models.items() : 
    with open(f"models/{name}_model.pkl", 'wb') as f :
        pickle.dump(model, f)
print("All models saved to models/folder successfully.")
# Final comparison summary
results = {
    "KNN" : round(accuracy_score(y_test, knn_pred) * 100, 2),
    "Naive Bayes" : round(accuracy_score(y_test, nb_pred) * 100, 2),
    "Logistic Regression" : round(accuracy_score(y_test, lr_pred) * 100, 2),
    "SVM" : round(accuracy_score(y_test, svm_pred) * 100, 2),
    "Random Forest" : round(accuracy_score(y_test, rf_pred) * 100, 2),
    "XGBoost" : round(accuracy_score(y_test, xgb_pred) * 100, 2),
}

for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {accuracy}%")