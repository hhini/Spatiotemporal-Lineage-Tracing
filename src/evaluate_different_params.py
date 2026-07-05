import numpy as np
import scipy.sparse as sp
import os
import sys
import scanpy as sc
from scipy.stats import energy_distance
from sklearn.neighbors import KernelDensity
import torch

# Add D:\001find\project_4_Spatiotemporal_Lineage\src\ to python path
sys.path.append(r"D:\001find\project_4_Spatiotemporal_Lineage\src")

from velocity_ode import load_spatial_dataset
from lineage_trace import DriftMLP, solve_ode_rk4, get_mean_shift_potential_fn

def main():
    real_prep_path = r"D:\001find\project_4_Spatiotemporal_Lineage\data\mosta_preprocessed.h5ad"
    holdout_model_path = r"D:\001find\project_4_Spatiotemporal_Lineage\data\drift_mlp_model_holdout.pt"
    
    adata = sc.read_h5ad(real_prep_path)
    adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
    adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
    adata_e115 = adata[adata.obs['timepoint'] == 'E11.5'].copy()
    
    coords_all = adata.obsm['spatial']
    coords_e95 = adata_e95.obsm['spatial']
    coords_e105 = adata_e105.obsm['spatial']
    coords_e115 = adata_e115.obsm['spatial']
    
    X_pca_95 = adata_e95.obsm['X_pca']
    state_95 = np.hstack([X_pca_95, coords_e95])
    
    input_dim = state_95.shape[1]
    model = DriftMLP(input_dim, hidden_dims=[128, 128, 64])
    model.load_state_dict(torch.load(holdout_model_path, map_location=torch.device('cpu')))
    model.eval()
    
    kde = KernelDensity(kernel='gaussian', bandwidth=30.0).fit(coords_all)
    log_dens_all = kde.score_samples(coords_all)
    dens_all = np.exp(log_dens_all)
    barrier_threshold = np.percentile(dens_all, 75)
    
    known_spatial = np.vstack([coords_e95, coords_e115])
    
    test_params = [
        (50.0, 0.2), # Original
        (40.0, 0.3),
        (30.0, 0.3),
        (30.0, 0.4),
        (20.0, 0.4),
        (10.0, 0.3)
    ]
    
    print(f"{'Sigma':<10}{'Eta':<10}{'Energy Distance':<20}{'BCR (%)':<10}")
    print("-" * 55)
    
    for sigma, eta in test_params:
        potential_fn = get_mean_shift_potential_fn(known_spatial, sigma=sigma, eta=eta)
        trajs = solve_ode_rk4(model, state_95, steps=10, potential_grad_fn=potential_fn)
        
        # ED
        predicted_spatial = trajs[5][:, -2:]
        ed = energy_distance(predicted_spatial[:, 0], coords_e105[:, 0]) + \
             energy_distance(predicted_spatial[:, 1], coords_e105[:, 1])
             
        # BCR
        cross_count = 0
        n_cells = state_95.shape[0]
        for i in range(n_cells):
            cell_crossed = False
            for step in range(1, 10):
                coord_t = trajs[step, i, -2:]
                log_dens = kde.score_samples(coord_t.reshape(1, -1))
                dens = np.exp(log_dens[0])
                if dens > barrier_threshold:
                    cell_crossed = True
                    break
            if cell_crossed:
                cross_count += 1
        bcr = (cross_count / n_cells) * 100.0
        
        print(f"{sigma:<10.1f}{eta:<10.2f}{ed:<20.6f}{bcr:<10.2f}")
        
    print("-" * 55)

if __name__ == "__main__":
    main()
