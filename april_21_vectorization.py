import numpy as np
import time
from scipy.spatial import distance_matrix

# 1. SETUP: Simulate a large tissue sample (2,000 cells)
n_cells = 2000
coords = np.random.rand(n_cells, 2) * 1000
labels = np.random.randint(0, 2, n_cells) # Random Type 0 or 1

# 2. THE OLD WAY (Loops)
start_loop = time.time()
loop_interactions = 0
# (We won't even run the full dist_mat logic here because it's so slow)
time_loop = time.time() - start_loop

# 3. THE SPACEIQ WAY (Vectorized)
start_vec = time.time()

# Step A: Get the adjacency matrix
dist_mat = distance_matrix(coords, coords)
adj = (dist_mat < 150) & (dist_mat > 0)

# Step B: Use a "Mask" to find where labels are different
# We create a grid where Grid[i,j] is True if Cell i and Cell j have different types
label_grid = labels[:, np.newaxis] != labels[np.newaxis, :]

# Step C: The "Magic" Step
# Count where they are neighbors AND different types simultaneously
vector_interactions = np.sum(adj & label_grid) / 2

time_vec = time.time() - start_vec

print(f"--- PERFORMANCE BENCHMARK ({n_cells} Cells) ---")
print(f"Vectorized Time: {time_vec:.4f} seconds")
print(f"Interactions Found: {int(vector_interactions)}")