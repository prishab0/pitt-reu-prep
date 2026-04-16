from skimage import data, filters, measure
from sklearn.cluster import KMeans
import numpy as np

# 1. Setup & Feature Extraction (April 9-11 logic)
image = data.coins()
binary = image > filters.threshold_otsu(image)
#labels = measure.label(binary)
#props = measure.regionprops(labels)

# CHANGE THIS LINE:
# labels = measure.label(binary)
# props = measure.regionprops(labels)

# TO THIS:
labels = measure.label(binary)
# We pass 'image' here so the computer can see the brightness/intensity
props = measure.regionprops(labels, intensity_image=image)

# 2. Build the Feature Matrix
# We collect Area and Intensity for every 'real' object
features = []
for p in props:
    if p.area > 100: # Filter dust
        features.append([p.area, p.mean_intensity])

features = np.array(features)

# 3. K-Means (Unbiased Typing)
# We tell the AI: "Find 2 distinct types of objects in this data"
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(features)

print("--- UNBIASED TYPING REPORT ---")
for i in range(2):
    count = np.sum(cluster_labels == i)
    print(f"Type {i}: Found {count} objects")
    
# Logic Check: Print the average area for each type
avg_area_0 = np.mean(features[cluster_labels == 0, 0])
avg_area_1 = np.mean(features[cluster_labels == 1, 0])
print(f"Avg Area Type 0: {avg_area_0:.2f}")
print(f"Avg Area Type 1: {avg_area_1:.2f}")