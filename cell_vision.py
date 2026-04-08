from skimage import data
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
image = data.coins()

# 2. DEFINE the variables first
dimensions = image.shape
pixel_val = image[50, 50] # This was the missing line!

# 3. PRINT (Now it works because variables are defined above)
print("--- DATA SUMMARY ---")
print(f"Image Dimensions: {dimensions}")
print(f"The brightness value at pixel (50,50) is: {pixel_val}")

# 4. VISUALIZE (We are making two plots side-by-side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: The actual image
ax1.imshow(image, cmap='gray')
ax1.set_title("Scientific View: Coins")

# Plot 2: The Histogram (The "Population" of pixels)
# .ravel() just flattens the 2D image into one long list of numbers
ax2.hist(image.ravel(), bins=256, range=(0, 255), color='gray')
ax2.set_title("Pixel Intensity Histogram")
ax2.set_xlabel("Brightness (0-255)")
ax2.set_ylabel("Number of Pixels")

plt.tight_layout()
plt.show()