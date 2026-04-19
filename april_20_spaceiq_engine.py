import numpy as np
from skimage import data, filters, measure
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

def run_spaceiq_pipeline(img, threshold_px=150):
    """
    The full pipeline wrapped into a single engine for batch processing.
    """
    try:
        # 1. Processing & Feature Extraction
        binary = img > filters.threshold_otsu(img)
        labels = measure.label(binary)
        props = measure.regionprops(labels, intensity_image=img)
        
        # FILTER GATE: Ensure we have enough cells to actually analyze
        valid_props = [p for p in props if p.area > 100]
        if len(valid_props) < 5:
            return None, "Error: Insufficient cell count"

        all_centroids = np.array([p.centroid for p in valid_props])
        features = np.array([[p.area, p.mean_intensity] for p in valid_props])

        # 2. Unbiased Typing
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)

        # 3. Spatial Interaction Math
        dist_mat = distance_matrix(all_centroids, all_centroids)
        adj_matrix = dist_mat < threshold_px
        
        total_conn = (np.sum(adj_matrix) - len(all_centroids)) / 2
        
        cross_type = 0
        for i in range(len(all_centroids)):
            for j in range(i + 1, len(all_centroids)):
                if adj_matrix[i, j] and cluster_labels[i] != cluster_labels[j]:
                    cross_type += 1

        score = (cross_type / total_conn) if total_conn > 0 else 0
        return score, "Success"

    except Exception as e:
        return None, f"Failure: {str(e)}"

# --- BATCH TEST SIMULATION ---
# Imagine these are 3 different patient samples
test_samples = [data.coins(), data.coins()[:, ::-1], data.coins()[::-1, :]]

print(f"{'Sample ID':<15} | {'Status':<15} | {'Mixing Score':<10}")
print("-" * 45)

for i, sample in enumerate(test_samples):
    score, msg = run_spaceiq_pipeline(sample)
    score_str = f"{score:.4f}" if score is not None else "N/A"
    print(f"Patient_{i:03}     | {msg:<15} | {score_str}")