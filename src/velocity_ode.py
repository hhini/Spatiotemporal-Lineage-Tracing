import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import shortest_path
import scanpy as sc
import anndata as ad
import yaml
import os

def load_spatial_dataset(adata_path, loom_path=None):
    """
    Loads the spatial transcriptomics AnnData and optionally merges it with RNA velocity loom data.
    If loom_path is not provided, computes Diffusion Pseudotime (DPT) and estimates velocity vectors.
    """
    print(f"[Prep] Loading spatial transcriptomics data from {adata_path}...")
    if not os.path.exists(adata_path):
        # Create a mock dataset for testing/bootstrap if the file doesn't exist yet
        print("[Prep] Adata path not found. Generating a mock dataset for testing...")
        n_cells = 500
        n_genes = 50
        X = np.random.negative_binomial(20, 0.3, size=(n_cells, n_genes))
        spatial = np.random.uniform(10, 500, size=(n_cells, 2))
        adata = ad.AnnData(X=X.astype(np.float32))
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm['spatial'] = spatial
        
        # Simulate simple differentiation pseudotime
        center = np.array([250.0, 250.0])
        dist_from_center = np.linalg.norm(spatial - center, axis=1)
        adata.obs['pseudotime'] = dist_from_center / np.max(dist_from_center)
    else:
        adata = sc.read_h5ad(adata_path)
        print(f"[Prep] Loaded raw dataset shape: {adata.shape}")
        
        # Balanced subsampling across developmental timepoints
        if adata.shape[0] > 4000:
            print("[Prep] Subsampling dataset to 4000 cells for memory efficiency...")
            np.random.seed(42)
            subsampled_indices = []
            for tp, count in [('E9.5', 1000), ('E10.5', 1500), ('E11.5', 1500)]:
                tp_indices = np.where(adata.obs['timepoint'] == tp)[0]
                if len(tp_indices) > count:
                    chosen = np.random.choice(tp_indices, size=count, replace=False)
                else:
                    chosen = tp_indices
                subsampled_indices.extend(chosen)
            adata = adata[sorted(subsampled_indices)].copy()
            print(f"[Prep] Subsampled dataset shape: {adata.shape}")

    # Compute PCA first
    if 'X_pca' not in adata.obsm:
        print("[Prep] Computing PCA (20 components)...")
        sc.tl.pca(adata, n_comps=20)

    # Initialize RNA velocity vector field
    if loom_path and os.path.exists(loom_path):
        print(f"[Prep] Splicing velocity loom found at {loom_path}. Merging data...")
        import scvelo as sv
        ldata = sv.read(loom_path, cache=True)
        adata = sv.utils.merge(adata, ldata)
        sv.tl.velocity(adata)
        sv.tl.velocity_graph(adata)
        if 'velocity' in adata.layers:
            adata.obsm['velocity_pca'] = adata.layers['velocity'][:, :20]
    else:
        print("[Prep] Estimating differentiation velocity vectors using Diffusion Pseudotime (DPT) gradient...")
        
        # Run neighbors, diffmap, and dpt to get robust pseudotime
        if 'dpt_pseudotime' not in adata.obs:
            print("[Prep] Computing Diffusion Pseudotime (DPT)...")
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
            sc.tl.diffmap(adata)
            # Find a root cell (an E9.5 cell if available)
            if 'timepoint' in adata.obs:
                e95_indices = np.where(adata.obs['timepoint'] == 'E9.5')[0]
                if len(e95_indices) > 0:
                    adata.uns['iroot'] = int(e95_indices[0])
                else:
                    adata.uns['iroot'] = 0
            else:
                adata.uns['iroot'] = 0
            sc.tl.dpt(adata)
        
        X_pca = adata.obsm['X_pca']
        pseudo = adata.obs['dpt_pseudotime'].values
        n_cells = adata.shape[0]
        
        # Build kNN in PCA space to find neighbors for gradient estimation
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=15).fit(X_pca)
        indices = nbrs.kneighbors(X_pca, return_distance=False)
        
        velocity_pca = np.zeros_like(X_pca)
        for i in range(n_cells):
            nbr_indices = indices[i]
            # Find neighbors with higher pseudotime
            better_nbrs = nbr_indices[pseudo[nbr_indices] > pseudo[i]]
            if len(better_nbrs) > 0:
                # Average direction in PCA space
                direction = np.mean(X_pca[better_nbrs] - X_pca[i], axis=0)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    velocity_pca[i] = direction / norm
            else:
                # Terminal cell: zero velocity
                velocity_pca[i] = np.zeros(X_pca.shape[1])
                
        adata.obsm['velocity_pca'] = velocity_pca
        print("[Prep] Differentiation velocity calculation completed.")
        
    return adata

def compute_morphology_impedance(spatial_coords, morphology_image=None, gamma=2.0):
    """
    Computes local physical impedance values for each cell coordinate.
    If no morphology image is provided, computes a Kernel Density Estimation (KDE)
    where high-density cell clusters act as physical impedance barriers (simulating tissue density).
    """
    n_cells = spatial_coords.shape[0]
    if morphology_image is not None:
        print("[Impedance] Extracting impedance values from provided morphology image...")
        # In real application, map spatial_coords to image coordinates and sample pixel intensity
        # Mock sampling:
        h, w = morphology_image.shape[:2]
        sampled_vals = []
        for x, y in spatial_coords:
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))
            sampled_vals.append(morphology_image[py, px])
        sampled_vals = np.array(sampled_vals)
    else:
        print("[Impedance] No morphology image provided. Estimating impedance via local cell density (KDE)...")
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth=30.0).fit(spatial_coords)
        log_dens = kde.score_samples(spatial_coords)
        sampled_vals = np.exp(log_dens)
        # Normalize to [0, 1]
        sampled_vals = (sampled_vals - sampled_vals.min()) / (sampled_vals.max() - sampled_vals.min() + 1e-8)
        
    # Apply exponential metric scaling: g(x) = exp(gamma * H(x))
    impedance = np.exp(gamma * sampled_vals)
    return impedance

def compute_geodesic_distance_matrix(spatial_coords, impedance_values, max_edge_length=50.0):
    """
    Builds a Delaunay spatial triangulation graph, weights edges by average physical impedance,
    and solves the all-pairs shortest paths using Dijkstra's algorithm.
    """
    print("[Geodesic] Constructing Delaunay triangulation graph...")
    tri = Delaunay(spatial_coords)
    n_cells = spatial_coords.shape[0]
    
    # Extract unique edges from triangulation
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                u, v = simplex[i], simplex[j]
                edges.add((min(u, v), max(u, v)))
                
    # Build sparse adjacency matrix with weighted edges
    row, col, data = [], [], []
    for u, v in edges:
        p1 = spatial_coords[u]
        p2 = spatial_coords[v]
        dist = np.linalg.norm(p1 - p2)
        
        # Only connect within max_edge_length to prevent spanning across empty tissue voids
        if dist <= max_edge_length:
            # Weight is physical distance multiplied by average local morphology impedance
            avg_imp = 0.5 * (impedance_values[u] + impedance_values[v])
            weight = dist * avg_imp
            
            row.append(u)
            col.append(v)
            data.append(weight)
            
            # Undirected graph
            row.append(v)
            col.append(u)
            data.append(weight)
            
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    print("[Geodesic] Solving all-pairs shortest paths using Dijkstra...")
    # Computes geodesic distances on the Riemannian tissue manifold
    geodesic_dist = shortest_path(csgraph=adj_matrix, method='D', directed=False)
    
    # Handle disconnected components by replacing inf values with a large penalization distance
    max_finite = np.max(geodesic_dist[np.isfinite(geodesic_dist)])
    geodesic_dist[~np.isfinite(geodesic_dist)] = max_finite * 2.0
    
    return geodesic_dist

if __name__ == "__main__":
    # Test script execution
    real_path = "data/mosta.h5ad"
    if os.path.exists(real_path):
        adata = load_spatial_dataset(real_path)
        out_prep_path = "data/mosta_preprocessed.h5ad"
        print(f"[Prep] Saving preprocessed dataset to {out_prep_path}...")
        adata.write_h5ad(out_prep_path)
    else:
        adata = load_spatial_dataset("")
    coords = adata.obsm['spatial']
    imp = compute_morphology_impedance(coords, gamma=2.0)
    geo_dist = compute_geodesic_distance_matrix(coords, imp, max_edge_length=100.0)
    print(f"Computed geodesic distance matrix of shape: {geo_dist.shape}")
    print(f"Min geodesic distance: {geo_dist.min():.2f}, Max geodesic distance: {geo_dist.max():.2f}")
