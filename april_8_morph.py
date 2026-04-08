from skimage import data, filters, morphology
import matplotlib.pyplot as plt

image = data.coins()
thresh = filters.threshold_otsu(image)
binary = image > thresh

# 1. Create our "Tool" (a disk with radius 3)
footprint = morphology.disk(3)

# 2. Apply Morphological Operations
eroded = morphology.binary_erosion(binary, footprint)
dilated = morphology.binary_dilation(binary, footprint)
opened = morphology.binary_opening(binary, footprint) # Erosion -> Dilation

# 3. Plot the results to compare
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(binary, cmap='gray')
axes[0].set_title("Original Binary")

axes[1].imshow(eroded, cmap='gray')
axes[1].set_title("Eroded (Shrunk)")

axes[2].imshow(opened, cmap='gray')
axes[2].set_title("Opened (Cleaned)")

plt.show()