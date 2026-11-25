import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Load Dataset

df = pd.read_csv("marketing_campaign.csv")

print("Dataset Preview:")
print(df.head())


# Data Cleaning

df = df.drop_duplicates()
df = df.dropna(subset=["Income"])
df = df[df["Year_Birth"] >= 1900]
df = df[df["Income"] < 600000]


# Select Features

features = [
    "Year_Birth", "Education", "Marital_Status", "Income",
    "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth"
]

X = df[features].copy()


# Convert Categorical Data

encoders = {}

for col in ["Education", "Marital_Status"]:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col].astype(str))


# Handle Missing Values

X = X.fillna(X.median())


# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for K-Means")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# K-Means Clustering

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)


# Evaluate Clustering

score = silhouette_score(X_scaled, labels)

print("\nK-Means Silhouette Score:")
print(round(score, 4))


# PCA for Visualization

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Cluster Visualization

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels,
    palette="Set1",
    s=45,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Customer Clusters")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()


# Cluster Distribution

plt.figure(figsize=(6, 4))

sns.countplot(
    x=labels,
    hue=labels,
    palette="Set1",
    legend=False
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("K-Means Cluster Distribution")
plt.xticks([0, 1], ["Cluster 0", "Cluster 1"])
plt.tight_layout()
plt.show()


# Cluster Comparison

print("\nCluster Comparison:")

profile = df[
    [
        "Income", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds"
    ]
].groupby(labels).mean().round(2)

print(profile)


# Save Clustered Data Separately
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Load Dataset

df = pd.read_csv("marketing_campaign.csv")

print("Dataset Preview:")
print(df.head())


# Data Cleaning

df = df.drop_duplicates()
df = df.dropna(subset=["Income"])
df = df[df["Year_Birth"] >= 1900]
df = df[df["Income"] < 600000]


# Select Features

features = [
    "Year_Birth", "Education", "Marital_Status", "Income",
    "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth"
]

X = df[features].copy()


# Convert Categorical Data

encoders = {}

for col in ["Education", "Marital_Status"]:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col].astype(str))


# Handle Missing Values

X = X.fillna(X.median())


# Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Elbow Method

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for K-Means")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# K-Means Clustering

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

labels = kmeans.fit_predict(X_scaled)


# Evaluate Clustering

score = silhouette_score(X_scaled, labels)

print("\nK-Means Silhouette Score:")
print(round(score, 4))


# PCA for Visualization

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Cluster Visualization

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels,
    palette="Set1",
    s=45,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Customer Clusters")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()


# Cluster Distribution

plt.figure(figsize=(6, 4))

sns.countplot(
    x=labels,
    hue=labels,
    palette="Set1",
    legend=False
)

plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("K-Means Cluster Distribution")
plt.xticks([0, 1], ["Cluster 0", "Cluster 1"])
plt.tight_layout()
plt.show()


# Cluster Comparison

print("\nCluster Comparison:")

profile = df[
    [
        "Income", "Recency", "MntWines", "MntFruits",
        "MntMeatProducts", "MntFishProducts",
        "MntSweetProducts", "MntGoldProds"
    ]
].groupby(labels).mean().round(2)

print(profile)


# Save Clustered Data Separately

clustered_data = df.copy()
clustered_data["KMeans_Cluster"] = labels

clustered_data.to_csv("customer_clustered_data.csv", index=False)


# Save Model and Preprocessing Objects

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(score, "silhouette_score.pkl")
joblib.dump(pca, "pca_model.pkl")

print("\nModel and preprocessing objects saved successfully!")
print("Clustered data saved as customer_clustered_data.csv")
clustered_data = df.copy()
clustered_data["KMeans_Cluster"] = labels

clustered_data.to_csv("customer_clustered_data.csv", index=False)


# Save Model and Preprocessing Objects

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(pca, "pca_model.pkl")

joblib.dump(features, "features.pkl")

print("\nModel and preprocessing objects saved successfully!")
print("Clustered data saved as customer_clustered_data.csv")