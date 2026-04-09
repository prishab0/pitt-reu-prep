from skimage import data, filters, measure, color
import matplotlib.pyplot as plt

# 1. Prepare the image
image = data.coins()
binary = image > filters.threshold_otsu(image)

# 2. LABEL the objects
# connectivity=2 means pixels touching diagonally count as the same object
labels = measure.label(binary, connectivity=2)

# 3. MEASURE the objects
props = measure.regionprops(labels)

print(f"--- BIOMETRIC REPORT ---")
print(f"Total objects found: {len(props)}")

# 4. Print stats for the first 3 objects
for i in range(3):
    print(f"Object {i+1}: Area = {props[i].area}px, Center = {props[i].centroid}")

# 5. Visualize with colors! 
# label2rgb gives each ID number a different color so we can see the separation
colored_labels = color.label2rgb(labels, image=image, bg_label=0)

plt.imshow(colored_labels)
plt.title("Labeled Objects (Each color is a unique ID)")
plt.show()