import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

header = []
header.append("label")
header.append("ex_label")

for i in range(14):
    header.append("Feature " + str(i + 1))

df1 = pd.read_csv("leaves.csv", header=None)
df1.columns = header

df2 = pd.read_csv('leaf_hog_lbp_features.csv')

df = pd.merge(df1, df2, on=['label', 'ex_label'])

y = df.pop("label")
y = np.array(y)
class_samples = df.pop("ex_label")

unique_labels = np.unique(y)

label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
y = np.vectorize(label_mapping.get)(y)

X = df

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=0.5)
# X_scaled = pca.fit_transform(X_scaled)

clustering_algorithms = [
    KMeans(n_clusters=30, random_state=42, n_init=20),
    AgglomerativeClustering(n_clusters=30),
    DBSCAN(eps=2, min_samples=5),
    GaussianMixture(n_components=30, random_state=42)
]

results = {}


def map_clusters_to_labels(y_true, y_pred):
    labels = np.unique(y_true)
    clusters = np.unique(y_pred)
    matrix = np.zeros((len(labels), len(clusters)))

    for i, label in enumerate(labels):
        for j, cluster in enumerate(clusters):
            matrix[i, j] = np.sum((y_true == label) & (y_pred == cluster))

    return np.argmax(matrix, axis=0)


for algorithm in clustering_algorithms:
    name = type(algorithm).__name__

    if name == 'GaussianMixture':
        algorithm.fit(X_scaled)
        y_pred = np.argmax(algorithm.predict_proba(X_scaled), axis=1)
    else:
        algorithm.fit(X_scaled)
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_
        else:
            y_pred = algorithm.predict(X_scaled)

    if len(np.unique(y_pred)) == 1:
        print(f'{name} did not find any clusters.')
        continue

    if len(np.unique(y_pred)) > 1:
        silhouette = silhouette_score(X_scaled, y_pred)
    else:
        silhouette = -1

    cluster_to_label_map = map_clusters_to_labels(y, y_pred)
    y_pred_mapped = np.array([cluster_to_label_map[cluster] for cluster in y_pred])

    conf_matrix = confusion_matrix(y, y_pred_mapped)
    accuracy = accuracy_score(y, y_pred_mapped)

    results[name] = {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'silhouette_score': silhouette
    }

for name, result in results.items():
    print(f'{name} - Accuracy: {result["accuracy"]}, Silhouette Score: {result["silhouette_score"]}')
    plt.figure(figsize=(10, 8))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
