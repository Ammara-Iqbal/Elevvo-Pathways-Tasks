import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib

def ensure_output_dir(path="outputs"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_data(path="Mall_Customers.csv"):
    """Load CSV and do light cleaning/renaming if necessary."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ['Annual Income (k$)', 'Spending Score (1-100)']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Found: {df.columns.tolist()}")
    return df


def basic_eda(df, outdir):
    print("--- Basic EDA ---")
    print(df.head())
    print(df.info())
    print(df.describe())
    print("Missing values per column:\n", df.isnull().sum())
    print("Duplicate rows:\n", df.duplicated().sum())

    df.head(50).to_csv(os.path.join(outdir, "sample_head.csv"), index=False)


def plot_distributions(df, outdir):
    # Histograms for numeric fields
    numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(6,4))
            plt.hist(df[col].dropna(), bins=20)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"dist_{col.replace(' ','_')}.png"))
            plt.close()

    # Scatter: Income vs Spending Score
    plt.figure(figsize=(6,5))
    plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Income vs Spending Score')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_income_spending.png"))
    plt.close()


def scale_features(df, feature_cols, scaler_type='standard'):
    """Return scaled numpy array and scaler object."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    X = df[feature_cols].values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k_elbow(X, k_range=range(1,11), outdir='outputs'):
    inertias = []
    Ks = list(k_range)
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(Ks, inertias, marker='o')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(Ks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'elbow_method.png'))
    plt.close()
    return Ks, inertias


def find_optimal_k_silhouette(X, k_range=range(2,11), outdir='outputs'):
    scores = {}
    Ks = []
    silhouette_vals = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        try:
            s = silhouette_score(X, labels)
        except Exception as e:
            s = np.nan
        Ks.append(k)
        silhouette_vals.append(s)
        scores[k] = s
    plt.figure(figsize=(6,4))
    plt.plot(Ks, silhouette_vals, marker='o')
    plt.xlabel('k (number of clusters)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for different k')
    plt.xticks(Ks)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'silhouette_scores.png'))
    plt.close()
    return scores


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def pca_2d(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def plot_clusters_pca(X_pca, labels, outpath, title='Clusters (PCA projection)'):
    plt.figure(figsize=(6,5))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(X_pca[mask,0], X_pca[mask,1], label=f'Cluster {lab}', alpha=0.7)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_income_vs_spending(df, labels, outpath):
    plt.figure(figsize=(6,5))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(df.loc[mask, 'Annual Income (k$)'], df.loc[mask, 'Spending Score (1-100)'], label=f'Cluster {lab}', alpha=0.7)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Income vs Spending Score (clustered)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_model(obj, path):
    joblib.dump(obj, path)


def main(csv_path='Mall_Customers.csv', outdir='outputs'):
    outdir = ensure_output_dir(outdir)

    # 1) Load data
    df = load_data(csv_path)

    # 2) Basic EDA and plots
    basic_eda(df, outdir)
    plot_distributions(df, outdir)

    # 3) Select features for clustering 
    feature_cols = ['Annual Income (k$)', 'Spending Score (1-100)']

    # 4) Scale features 
    X_scaled, scaler = scale_features(df, feature_cols, scaler_type='standard')

    # 5) Use elbow method and silhouette to pick k
    Ks, inertias = find_optimal_k_elbow(X_scaled, k_range=range(1,11), outdir=outdir)
    silhouette_scores = find_optimal_k_silhouette(X_scaled, k_range=range(2,11), outdir=outdir)
    print('\nElbow inertias:', dict(zip(Ks, inertias)))
    print('\nSilhouette scores:', silhouette_scores)

    # Choose k 
    chosen_k = 5
    print(f"\nUsing chosen_k = {chosen_k} (change this after inspecting plots)")

    # 6) Fit KMeans and attach labels to dataframe
    kmeans, labels = fit_kmeans(X_scaled, chosen_k)
    df['Cluster'] = labels

    # Save clustered CSV
    df.to_csv(os.path.join(outdir, 'mall_customers_with_clusters.csv'), index=False)

    # 7) Examine cluster centroids (in original feature units)
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroids_df.index.name = 'Cluster'
    centroids_df.to_csv(os.path.join(outdir, 'cluster_centroids.csv'))
    print('\nCluster centroids (original units):\n', centroids_df)

    # 8) Cluster profiles
    profile = df.groupby('Cluster')[['Age'] + feature_cols].agg(['count', 'mean', 'median', 'std'])
    profile.to_csv(os.path.join(outdir, 'cluster_profile.csv'))
    print('\nCluster profile saved to outputs/cluster_profile.csv')

    # 9) PCA for visualization and plotting
    pca, X_pca = pca_2d(X_scaled, n_components=2)
    plot_clusters_pca(X_pca, labels, os.path.join(outdir, 'clusters_pca.png'))
    plot_income_vs_spending(df, labels, os.path.join(outdir, 'income_vs_spending_clustered.png'))

    # 10) Save models (scaler + kmeans + pca)
    save_model(scaler, os.path.join(outdir, 'scaler.joblib'))
    save_model(kmeans, os.path.join(outdir, 'kmeans.joblib'))
    save_model(pca, os.path.join(outdir, 'pca.joblib'))

    # 11) Evaluation: silhouette score for the chosen k
    try:
        sil = silhouette_score(X_scaled, labels)
    except Exception:
        sil = np.nan
    print(f"Silhouette score for k={chosen_k}: {sil}")

    # 12) Bonus: DBSCAN 
    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    df['DBSCAN_Cluster'] = db_labels
    df.to_csv(os.path.join(outdir, 'mall_customers_dbscan_example.csv'), index=False)
    plt.figure(figsize=(6,5))
    unique_db = np.unique(db_labels)
    for lab in unique_db:
        mask = db_labels == lab
        marker = 'x' if lab == -1 else 'o'
        plt.scatter(df.loc[mask,'Annual Income (k$)'], df.loc[mask,'Spending Score (1-100)'], label=f'Cluster {lab}', marker=marker, alpha=0.7)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('DBSCAN clustering (example)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'dbscan_income_spending.png'))
    plt.close()

    print('\nAll outputs saved to the "{}" folder. Inspect the elbow and silhouette plots to choose a different k if you like.'.format(outdir))
    print('Files saved: mall_customers_with_clusters.csv, cluster_centroids.csv, cluster_profile.csv, elbow_method.png, silhouette_scores.png, clusters_pca.png, income_vs_spending_clustered.png, dbscan_income_spending.png, scaler.joblib, kmeans.joblib, pca.joblib')


if __name__ == '__main__':
    main(csv_path='Mall_Customers.csv', outdir='outputs')
