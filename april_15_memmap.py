import numpy as np
import os
import time

# 1. Setup: Define a "Giant" dummy biopsy (100 Million Pixels)
filename = "giant_biopsy.dat"
shape = (10000, 10000) 

# 2. WRITE: Create the file on the hard drive (not in RAM)
print("Creating 1GB file on disk...")
# 'w+' means write/create mode
fp = np.memmap(filename, dtype='float32', mode='w+', shape=shape)

# Fill it with random 'cell' data
fp[:] = np.random.rand(*shape)[:]
fp.flush() # Secure the data to the disk
print("File created successfully.")

# 3. READ: The 'Scaling' Hack
# We open it in 'read' mode. This uses NEARLY ZERO RAM.
new_fp = np.memmap(filename, dtype='float32', mode='r', shape=shape)

# Access a 100x100 'neighborhood' in the very center
start_time = time.time()
neighborhood = new_fp[5000:5100, 5000:5100]
duration = time.time() - start_time

print(f"\n--- SCALABILITY REPORT ---")
print(f"Neighborhood Slice extracted in: {duration:.4f} seconds")
print(f"Mean intensity of slice: {neighborhood.mean():.4f}")

# 4. Cleanup: Remove the dummy file
os.remove(filename)