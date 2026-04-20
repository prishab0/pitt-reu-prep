import numpy as np

def calculate_mixing_score(adj_matrix, labels):
    """
    Calculates the spatial mixing score between two cell types.

    Args:
        adj_matrix (np.ndarray): An NxN boolean adjacency matrix where 
            True indicates cells are within the interaction threshold.
        labels (np.ndarray): An array of length N containing cluster 
            labels (0 or 1) for each cell.

    Returns:
        float: The mixing score (0.0 to 1.0), where 1.0 represents 
            perfect inter-type mixing and 0.0 represents complete separation.
            
    Performance Note: 
        This function uses vectorized matrix broadcasting for O(N) efficiency.
    """
    # Remove self-connections (diagonal)
    n = len(labels)
    adj = adj_matrix & ~np.eye(n, dtype=bool)
    
    total_connections = np.sum(adj) / 2
    
    if total_connections == 0:
        return 0.0
        
    # Vectorized check for label differences
    label_mismatch = labels[:, np.newaxis] != labels[np.newaxis, :]
    
    # Count interactions where neighbor=True AND labels=Different
    cross_interactions = np.sum(adj & label_mismatch) / 2
    
    return cross_interactions / total_connections

# Example usage for verification
print(help(calculate_mixing_score))