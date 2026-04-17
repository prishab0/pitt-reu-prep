import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, measure
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

# 1. DATA PIPELINE (The "SpaceIQ" Backend)
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels, intensity_image=image)

# Extract spatial coordinates (centroids) and features for clustering
all_centroids = []
features = []
for p in props:
    if p.area > 100:
        all_centroids.append(p.centroid)
        features.append([p.area, p.mean_intensity])

all_centroids = np.array(all_centroids)
features = np.array(features)

# 2. UNBIASED TYPING (Clustering)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

# 3. SPATIAL GRAPH (Adjacency Matrix)
dist_mat = distance_matrix(all_centroids, all_centroids)
interaction_threshold = 150
adj_matrix = dist_mat < interaction_threshold

# 4. VISUALIZATION (The "Trust" Layer)
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(image, cmap='gray')

# Plot the Cells as colored nodes (Blue = Type 0, Red = Type 1)
colors = ['blue' if label == 0 else 'red' for label in cluster_labels]
ax.scatter(all_centroids[:, 1], all_centroids[:, 0], c=colors, s=30, edgecolors='white', zorder=3)

# Draw the interaction lines (The "Yellow Battle Lines")
print("Drawing interaction lines...")
for i in range(len(all_centroids)):
    for j in range(i + 1, len(all_centroids)):
        # ONLY draw a line if they are close AND different types
        if adj_matrix[i, j] and cluster_labels[i] != cluster_labels[j]:
            ax.plot([all_centroids[i, 1], all_centroids[j, 1]], 
                    [all_centroids[i, 0], all_centroids[j, 0]], 
                    color='yellow', linewidth=1.5, alpha=0.7, zorder=2)

ax.set_title("SpaceIQ Interaction Map: Unbiased Cell-Type Mixing")
plt.axis('off') # Cleaner look for the presentation
plt.show()