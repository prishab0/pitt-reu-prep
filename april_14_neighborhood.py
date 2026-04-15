import numpy as np
from skimage import data, filters, measure

# 1. Setup (Using our filtered logic from April 11)
image = data.coins()
binary = image > filters.threshold_otsu(image)
labels = measure.label(binary)
props = measure.regionprops(labels)

# 2. Get all centroids into one big NumPy Vector (The GPU Mindset)
all_centroids = np.array([p.centroid for p in props if p.area > 100])

# 3. Pick one "Target" cell (let's say the first one in the list)
target_cell = all_centroids[0]

# 4. VECTORIZED TRANSFORMATION
# Subtract the target from EVERY other cell at the same time
local_coordinates = all_centroids - target_cell

print(f"--- SPACEIQ NEIGHBORHOOD MAP ---")
print(f"Target Cell Global Position: {target_cell}")
print(f"First 5 Neighbors (Local Offset):")
print(local_coordinates[1:6]) # Skip 0 because that's the target itself (0,0)