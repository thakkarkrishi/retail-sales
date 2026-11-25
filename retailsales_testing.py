import pandas as pd
import joblib


# Load Saved Model

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")
score = joblib.load("silhouette_score.pkl")


# New Customer Data (Without ID and Cluster)

# New Customer Data

# New Customer Data

new_data = {
    "Year_Birth": 1972,
    "Education": "PhD",
    "Marital_Status": "Divorced",
    "Income": 85000.0,
    "Kidhome": 0,
    "Teenhome": 0,
    "Recency": 10,
    "MntWines": 1100,
    "MntFruits": 120,
    "MntMeatProducts": 900,
    "MntFishProducts": 250,
    "MntSweetProducts": 180,
    "MntGoldProds": 150,
    "NumDealsPurchases": 1,
    "NumWebPurchases": 8,
    "NumCatalogPurchases": 9,
    "NumStorePurchases": 10,
    "NumWebVisitsMonth": 2
}


# Convert Data into DataFrame

new_customer = pd.DataFrame([new_data])


# Select Features

X_new = new_customer[features].copy()


# Convert Categorical Data

for col in ["Education", "Marital_Status"]:
    X_new[col] = encoders[col].transform(X_new[col].astype(str))


# Feature Scaling

X_new_scaled = scaler.transform(X_new)


# Predict Cluster

prediction = kmeans.predict(X_new_scaled)


# Display Prediction

print("\n========== CUSTOMER PREDICTION ==========")

print("Predicted Cluster:", prediction[0])

if prediction[0] == 0:
    print("Customer belongs to Cluster 0")
else:
    print("Customer belongs to Cluster 1")


# Display Model Evaluation

print("\n========== MODEL EVALUATION ==========")

print("Silhouette Score:", round(score, 4))
print("Number of Clusters:", kmeans.n_clusters)