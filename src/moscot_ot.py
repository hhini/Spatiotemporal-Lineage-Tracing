import numpy as np
import scipy.sparse as sp
import yaml
import os
import anndata as ad
from velocity_ode import load_spatial_dataset, compute_morphology_impedance, compute_geodesic_distance_matrix

def compute_velocity_guided_cost(X_t, X_tp1, V_t, beta=0.5, eta=1e-6):
    """
    Computes the directed, asymmetric expression cost matrix.
    C[i, j] penalizes transitions from cell i (at time t) to cell j (at time t+1)
    if the displacement vector (X_tp1[j] - X_t[i]) goes against cell i's RNA velocity V_t[i].
    """
    n_t = X_t.shape[0]
    n_tp1 = X_tp1.shape[0]
    
    # 1. Compute baseline expression distance (1 - Cosine Similarity)
    print("[Cost] Computing baseline expression cost matrix...")
    X_t_norm = X_t / (np.linalg.norm(X_t, axis=1, keepdims=True) + eta)
    X_tp1_norm = X_tp1 / (np.linalg.norm(X_tp1, axis=1, keepdims=True) + eta)
    C_base = 1.0 - np.dot(X_t_norm, X_tp1_norm.T)
    
    # 2. Compute velocity drift penalty
    print("[Cost] Computing RNA velocity drift penalty...")
    # Displacement matrix of shape (n_t, n_tp1, D)
    # To prevent OOM for large datasets, we compute this in batches
    D_dim = X_t.shape[1]
    penalty = np.zeros((n_t, n_tp1))
    
    # Normalize velocity vectors
    V_t_norm = V_t / (np.linalg.norm(V_t, axis=1, keepdims=True) + eta)
    
    batch_size = 500
    for start_i in range(0, n_t, batch_size):
        end_i = min(start_i + batch_size, n_t)
        # shape: (batch, 1, D)
        X_t_batch = X_t[start_i:end_i, np.newaxis, :] 
        # shape: (1, n_tp1, D)
        X_tp1_all = X_tp1[np.newaxis, :, :] 
        
        # Displacement vectors: shape (batch, n_tp1, D)
        displacement = X_tp1_all - X_t_batch
        disp_norm = np.linalg.norm(displacement, axis=2, keepdims=True) + eta
        disp_dir = displacement / disp_norm
        
        # Dot product with velocity directions: shape (batch, n_tp1)
        # V_t_norm[start_i:end_i] has shape (batch, D) -> we make it (batch, 1, D)
        v_dir_batch = V_t[start_i:end_i, np.newaxis, :]
        v_norm = np.linalg.norm(v_dir_batch, axis=2, keepdims=True) + eta
        v_dir_batch = v_dir_batch / v_norm
        
        cos_theta = np.sum(disp_dir * v_dir_batch, axis=2) # shape (batch, n_tp1)
        
        # If velocity is zero (terminal cell), penalty should be zero
        no_vel = (np.linalg.norm(V_t[start_i:end_i], axis=1) == 0)
        
        # Penalty is (1 - cos_theta)
        batch_penalty = 1.0 - cos_theta
        batch_penalty[no_vel, :] = 0.0
        
        penalty[start_i:end_i, :] = batch_penalty
        
    C_expr = C_base + beta * penalty
    return C_expr

def entropic_sinkhorn(C, epsilon, p, q, max_iter=1000, tol=1e-4):
    """
    Solves the standard entropic optimal transport problem (Sinkhorn-Knopp algorithm).
    """
    # Kernel matrix
    K = np.exp(-C / epsilon)
    # Avoid numerical underflow
    K = np.clip(K, 1e-300, None)
    
    u = np.ones(C.shape[0]) / C.shape[0]
    v = np.ones(C.shape[1]) / C.shape[1]
    
    for iteration in range(max_iter):
        u_prev = u.copy()
        
        # Update scaling vectors
        u = p / (np.dot(K, v) + 1e-16)
        v = q / (np.dot(K.T, u) + 1e-16)
        
        # Check convergence
        if iteration % 10 == 0:
            err = np.linalg.norm(u - u_prev) / (np.linalg.norm(u) + 1e-16)
            if err < tol:
                break
                
    # Return coupling matrix (transport plan)
    pi = np.dot(np.diag(u), np.dot(K, np.diag(v)))
    return pi

def fused_gromov_wasserstein(C_expr, D_t, D_tp1, p, q, alpha=0.3, epsilon=0.05, max_iter=50, tol=1e-4):
    """
    Solves the Fused Gromov-Wasserstein optimal transport problem.
    Reconciles expression cost C_expr and spatial topological distances D_t, D_tp1.
    """
    n_t, n_tp1 = C_expr.shape
    
    # Initialize transport plan pi as the product of marginals (independent plan)
    pi = np.outer(p, q)
    
    # Precompute squared distance matrices for Gromov-Wasserstein constant terms
    D_t_sq = D_t ** 2
    D_tp1_sq = D_tp1 ** 2
    
    print("[FGW] Iterating Fused Gromov-Wasserstein solver...")
    for iteration in range(max_iter):
        pi_prev = pi.copy()
        
        # Compute Gromov-Wasserstein cost term:
        # L(pi) = (1 - alpha)*C_expr + alpha * (D_t_sq * p * q^T + p * q^T * D_tp1_sq^T - 2 * D_t * pi * D_tp1^T)
        # Note: D_t_sq * p is shape (n_t, 1), multiplying by q^T (1, n_tp1) gives (n_t, n_tp1)
        term1 = np.outer(np.dot(D_t_sq, p), q)
        term2 = np.outer(p, np.dot(D_tp1_sq, q))
        term3 = 2.0 * np.dot(D_t, np.dot(pi, D_tp1.T))
        
        C_gw = term1 + term2 - term3
        
        # Combined cost
        C_total = (1.0 - alpha) * C_expr + alpha * C_gw
        
        # Solve inner Entropic OT problem
        pi = entropic_sinkhorn(C_total, epsilon, p, q, max_iter=300, tol=1e-4)
        
        # Check convergence
        diff = np.linalg.norm(pi - pi_prev) / (np.linalg.norm(pi) + 1e-16)
        if diff < tol:
            print(f"[FGW] Converged at iteration {iteration}.")
            break
            
    return pi

def run_spatiotemporal_ot(adata_t, adata_tp1, config_path):
    """
    Computes spatial geodesic distances, constructs the velocity-guided cost,
    and solves the spatiotemporal FGW coupling.
    """
    # Load parameters from configuration yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    epsilon = float(config['ot_params']['epsilon'])
    alpha = float(config['ot_params']['alpha'])
    max_iter = int(config['ot_params']['max_iterations'])
    tol = float(config['ot_params']['tolerance'])
    
    beta = float(config['vg_params']['beta'])
    eta = float(config['vg_params']['eta'])
    
    gamma = float(config['geodesic_params']['gamma'])
    delaunay_max_edge_length = float(config['geodesic_params']['delaunay_max_edge_length'])
    
    # 1. Get spatial coordinates
    coords_t = adata_t.obsm['spatial']
    coords_tp1 = adata_tp1.obsm['spatial']
    
    # 2. Get PCA embeddings
    X_t = adata_t.obsm['X_pca']
    X_tp1 = adata_tp1.obsm['X_pca']
    V_t = adata_t.obsm['velocity_pca']
    
    # 3. Compute morphology impedance & geodesic distances
    print("[OT Pipeline] Computing physical geodesic distances at time t...")
    imp_t = compute_morphology_impedance(coords_t, gamma=gamma)
    D_t = compute_geodesic_distance_matrix(coords_t, imp_t, max_edge_length=delaunay_max_edge_length)
    D_t = D_t / (np.mean(D_t) + 1e-12)
    
    print("[OT Pipeline] Computing physical geodesic distances at time t+1...")
    imp_tp1 = compute_morphology_impedance(coords_tp1, gamma=gamma)
    D_tp1 = compute_geodesic_distance_matrix(coords_tp1, imp_tp1, max_edge_length=delaunay_max_edge_length)
    D_tp1 = D_tp1 / (np.mean(D_tp1) + 1e-12)
    
    # 4. Compute Velocity-Guided Cost matrix
    C_expr = compute_velocity_guided_cost(X_t, X_tp1, V_t, beta=beta, eta=eta)
    
    # 5. Define cell-level marginal uniform priors (can be adjusted by growth rate)
    p = np.ones(adata_t.shape[0]) / adata_t.shape[0]
    q = np.ones(adata_tp1.shape[0]) / adata_tp1.shape[0]
    
    # 6. Solve Fused Gromov-Wasserstein
    pi = fused_gromov_wasserstein(
        C_expr, D_t, D_tp1, p, q,
        alpha=alpha,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol
    )
    
    return pi

if __name__ == "__main__":
    import os
    real_prep_path = "data/mosta_preprocessed.h5ad"
    config_path = "config/moscot_config.yaml"
    
    if os.path.exists(real_prep_path):
        import scanpy as sc
        print(f"[OT Pipeline] Loading preprocessed dataset from {real_prep_path}...")
        adata = sc.read_h5ad(real_prep_path)
        
        # Split by timepoint
        print("[OT Pipeline] Splitting dataset by timepoints...")
        adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
        adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
        adata_e115 = adata[adata.obs['timepoint'] == 'E11.5'].copy()
        
        print(f"  E9.5 cells: {adata_e95.shape[0]}")
        print(f"  E10.5 cells: {adata_e105.shape[0]}")
        print(f"  E11.5 cells: {adata_e115.shape[0]}")
        
        # Solve E9.5 -> E10.5 optimal transport
        print("[OT Pipeline] Solving optimal transport for E9.5 -> E10.5...")
        pi_95_105 = run_spatiotemporal_ot(adata_e95, adata_e105, config_path)
        np.save("data/pi_95_105.npy", pi_95_105)
        print(f"  Saved pi_95_105 of shape {pi_95_105.shape}")
        
        # Solve E10.5 -> E11.5 optimal transport
        print("[OT Pipeline] Solving optimal transport for E10.5 -> E11.5...")
        pi_105_115 = run_spatiotemporal_ot(adata_e105, adata_e115, config_path)
        np.save("data/pi_105_115.npy", pi_105_115)
        print(f"  Saved pi_105_115 of shape {pi_105_115.shape}")
        
        # Solve E9.5 -> E11.5 optimal transport (Hold-out validation)
        print("[OT Pipeline] Solving optimal transport for E9.5 -> E11.5 (Hold-out)...")
        pi_95_115 = run_spatiotemporal_ot(adata_e95, adata_e115, config_path)
        np.save("data/pi_95_115.npy", pi_95_115)
        print(f"  Saved pi_95_115 of shape {pi_95_115.shape}")
        
    else:
        # Test script execution using dummy dataset
        print("[Test] Creating dummy datasets...")
        adata_t = load_spatial_dataset("")
        adata_tp1 = load_spatial_dataset("")
        pi = run_spatiotemporal_ot(adata_t, adata_tp1, config_path)
        print(f"Computed OT coupling matrix of shape: {pi.shape}")
        print(f"Coupling sum: {pi.sum():.6f} (Expected: 1.0)")
        print(f"Marginal t discrepancy: {np.linalg.norm(pi.sum(axis=1) - 1.0/adata_t.shape[0]):.6f}")
