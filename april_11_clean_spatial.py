from skimage import data, filters, measure
import numpy as np

# 1. Setup
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels)

img_center = np.array(image.shape) / 2

# 2. NEW: Filtered Spatial Analysis
clean_distances = []
clean_ids = []

for p in props:
    # Only process objects with an area larger than 100 pixels
    # This ignores the 'dust' that created ID #78 in your last run
    if p.area > 100:
        dist = np.linalg.norm(np.array(p.centroid) - img_center)
        clean_distances.append(dist)
        clean_ids.append(p.label) # Save the real ID

# 3. Results
if clean_distances:
    closest_idx = np.argmin(clean_distances)
    real_id = clean_ids[closest_idx]
    
    print(f"--- CLEAN SPATIAL REPORT ---")
    print(f"Total real objects found (Area > 100): {len(clean_ids)}")
    print(f"The REAL coin closest to center is ID #{real_id}")
    print(f"Distance: {clean_distances[closest_idx]:.2f} pixels")
else:
    print("No objects found matching the size criteria.")