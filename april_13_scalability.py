import numpy as np
import time

# Create a massive 'biopsy' of 1 million data points
data = np.random.rand(1000000)

# --- METHOD A: THE SLOW LOOP ---
start_time = time.time()
slow_result = []
for x in data:
    slow_result.append(x * 2)
loop_duration = time.time() - start_time
print(f"Slow Loop Time: {loop_duration:.4f} seconds")

# --- METHOD B: VECTORIZATION (The GPU Mindset) ---
start_time = time.time()
fast_result = data * 2 # This happens 'all at once' in memory
vector_duration = time.time() - start_time
print(f"Vectorized Time: {vector_duration:.4f} seconds")

print(f"\nSpeedup: {loop_duration / vector_duration:.1f}x faster!")