from skimage import data, filters, measure
import numpy as np

# 1. Setup
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels)

# 2. Find the center of the whole image
img_center = np.array(image.shape) / 2
print(f"Image Center: {img_center}")

# 3. Spatial Analysis: Which coin is closest to the center?
distances = []
for p in props:
    # Calculate distance from this coin's center to image center
    dist = np.linalg.norm(np.array(p.centroid) - img_center)
    distances.append(dist)

# 4. Results
closest_id = np.argmin(distances)
print(f"--- SPATIAL REPORT ---")
print(f"The coin closest to the center is ID #{closest_id}")
print(f"It is only {distances[closest_id]:.2f} pixels away from the middle.")