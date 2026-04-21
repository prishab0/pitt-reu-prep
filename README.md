# SpaceIQ: Unbiased Spatial Interaction Engine

This repository contains a high-performance Python pipeline for analyzing spatial cell-type interactions in multiplexed tissue imaging. Developed as part of the preparation for the **TECBio REU** and **PredXBio** research internship at the University of Pittsburgh.

## 🔬 Biological Context
In tumor microenvironments, the proximity between different cell types (e.g., Immune vs. Cancer) is a critical biomarker for treatment response. This engine identifies:
* **"Hot" Tumors:** High spatial mixing between cell types, suggesting active immune infiltration.
* **"Cold" Tumors:** Clear spatial separation or exclusion, where immune cells are kept at the periphery.

## 🚀 Key Features
* **Unbiased Cell Typing:** Uses K-Means clustering on morphological (area) and intensity features to identify cell populations without prior labeling.
* **Spatial Graphing:** Builds a neighborhood map using an adjacency matrix based on a tunable interaction radius (default 150px).
* **High-Performance Vectorization:** Eliminates slow Python loops in favor of NumPy broadcasting, enabling the analysis of millions of cells in seconds.
* **Trust-Layer Visualization:** Overlays spatial interaction "battle lines" directly onto tissue images to verify algorithm accuracy.

## 💻 Usage
```python
from april_20_spaceiq_engine import run_spaceiq_pipeline

# Analyze a biopsy image and get the Mixing Score
mixing_score, status = run_spaceiq_pipeline(image_data, threshold_px=150)

print(f"Spatial mixing score: {mixing_score:.4f}")
print(f"Infiltration Status: {status}")
📊 Performance BenchmarksTested on Prisha's MacBook Air (M-Series Unified Memory):2,000 Cells: ~0.004 seconds (Vectorized logic)Scalability: $O(N)$ logic designed for GPU acceleration via CuPy or PyTorch.