
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "monthly_income",
    "monthly_spending",
    "savings_balance",
    "existing_debt",
    "credit_util",
    "debt_to_income",
    "savings_to_income",
]


class Km:

    def __init__(self):
        self.path = DATA_DIR / "loans_merged.csv"

    @staticmethod
    def load_dataset(path):
        return pd.read_csv(path)

    def train_and_save(self):

        
        df = self.load_dataset(self.path)

         
        missing_columns = [
            col for col in FEATURES if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

       
        X = df[FEATURES].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()

        if len(X) < 10:
            raise ValueError(
                "Not enough valid rows to train K-Means."
            )

        print(f"Dataset shape for clustering: {X.shape}")

    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        
        inertias = []
        k_values = range(2, 11)

        for k in k_values:
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42
            )

            model.fit(X_scaled)
            inertias.append(model.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(
            list(k_values),
            inertias,
            marker="o"
        )

        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal K")
        plt.xticks(list(k_values))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        elbow_path = MODELS_DIR / "kmeans_elbow.png"
        plt.savefig(elbow_path, dpi=300)
        print(f"Elbow plot saved to {elbow_path}")
        plt.show()
        plt.close()

         
        optimal_k = 3

        final_model = KMeans(
            n_clusters=optimal_k,
            init="k-means++",
            n_init=10,
            random_state=42
        )

        labels = final_model.fit_predict(X_scaled)


        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(2000, len(X_scaled)),
            random_state=42
        )

        print(f"Silhouette Score: {score:.4f}")

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        cluster_sizes = dict(zip(
            unique_labels.tolist(),
            counts.tolist()
        ))

        print(f"Cluster sizes: {cluster_sizes}")

     
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        centroids_pca = pca.transform(
            final_model.cluster_centers_
        )

        plt.figure(figsize=(10, 7))

        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,
            cmap="viridis",
            s=15,
            alpha=0.6
        )

        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            marker="X",
            s=250,
            c="red",
            edgecolors="black",
            label="Centroids"
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(
            f"K-Means Customer Segmentation (K={optimal_k})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        pca_path = MODELS_DIR / "kmeans_clusters_pca.png"
        plt.savefig(pca_path, dpi=300)
        print(f"PCA cluster plot saved to {pca_path}")
        plt.show()
        plt.close()

        explained_variance = (
            pca.explained_variance_ratio_.sum() * 100
        )

        print(
            "Variance explained by 2 PCA components: "
            f"{explained_variance:.2f}%"
        )
 
        joblib.dump(
            final_model,
            MODELS_DIR / "kmeans.pkl"
        )

        joblib.dump(
            scaler,
            MODELS_DIR / "kmeans_scaler.pkl"
        )

        print("K-Means model saved.")
        print("Scaler saved.")
   
        cluster_data = X.copy()
        cluster_data["cluster"] = labels

       
        profile = cluster_data.groupby("cluster")[FEATURES].mean().round(2)

        print("\nCluster Profiles:")
        print(profile.to_string())

      
        profile.to_csv(MODELS_DIR / "kmeans_cluster_profiles.csv")
        cluster_data.to_csv(
            MODELS_DIR / "kmeans_cluster_assignments.csv",
            index=False
        )

        print("\nCluster profiles and assignments saved.")

        return final_model, labels, final_model.cluster_centers_


def main():
    model = Km()
    model.train_and_save()


if __name__ == "__main__":
    main()
