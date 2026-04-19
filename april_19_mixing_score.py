import numpy as np
from skimage import data, filters, measure
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

# 1. PIPELINE RECAP (The Backend)
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels, intensity_image=image)

all_centroids = np.array([p.centroid for p in props if p.area > 100])
features = np.array([[p.area, p.mean_intensity] for p in props if p.area > 100])

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

dist_mat = distance_matrix(all_centroids, all_centroids)
adj_matrix = dist_mat < 150 # 150 pixel interaction zone

# 2. CALCULATE THE SCORE
# Subtract length of centroids because dist_mat includes the distance from a cell to itself (0)
total_connections = (np.sum(adj_matrix) - len(all_centroids)) / 2

# Count cross-type interactions
cross_type_interactions = 0
for i in range(len(all_centroids)):
    for j in range(i + 1, len(all_centroids)):
        if adj_matrix[i, j] and cluster_labels[i] != cluster_labels[j]:
            cross_type_interactions += 1

mixing_score = (cross_type_interactions / total_connections) if total_connections > 0 else 0

# 3. CLINICAL SUMMARY
print(f"--- SPACEIQ SPATIAL BIOMARKER REPORT ---")
print(f"Total Cells Analyzed: {len(all_centroids)}")
print(f"Total Neighborhood Connections: {int(total_connections)}")
print(f"Cross-Type Interactions: {cross_type_interactions}")
print(f"FINAL MIXING SCORE: {mixing_score:.4f}")

if mixing_score > 0.5:
    print("STATUS: 'Hot' Tumor - High Immune Infiltration Predicted.")
else:
    print("STATUS: 'Cold' Tumor - Potential Exclusion of Immune Cells.")