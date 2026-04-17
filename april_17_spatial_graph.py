import numpy as np
from skimage import data, filters, measure
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

# 1. SETUP: Process the image to get centroids and clusters
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels, intensity_image=image)

# 2. EXTRACT FEATURES: Get Centroids and Area/Intensity
all_centroids = []
features = []

for p in props:
    if p.area > 100:
        all_centroids.append(p.centroid)
        features.append([p.area, p.mean_intensity])

all_centroids = np.array(all_centroids)
features = np.array(features)

# 3. CLUSTER: Create the unbiased labels (Type 0 vs Type 1)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# --- NOW THE APRIL 17 CODE WILL WORK ---

# 4. Calculate the Distance Matrix
dist_mat = distance_matrix(all_centroids, all_centroids)

# 5. Define the "Interaction Zone" (150 pixels)
interaction_threshold = 150
adj_matrix = dist_mat < interaction_threshold

# 6. Find Cross-Type Interactions
interactions = 0
for i in range(len(all_centroids)):
    for j in range(i + 1, len(all_centroids)):
        if adj_matrix[i, j]:
            if cluster_labels[i] != cluster_labels[j]:
                interactions += 1

print(f"--- SPATIAL INTERACTION REPORT ---")
print(f"Total Neighborhood Connections: {np.sum(adj_matrix) - len(all_centroids)}")
print(f"Cross-Type Interactions (Type 0 <-> Type 1): {interactions}")