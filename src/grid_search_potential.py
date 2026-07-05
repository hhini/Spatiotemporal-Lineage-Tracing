import numpy as np
import scipy.sparse as sp
import os
import sys
import scanpy as sc
from scipy.stats import energy_distance
import torch

# Add D:\001find\project_4_Spatiotemporal_Lineage\src to python path
sys.path.append(r"D:\001find\project_4_Spatiotemporal_Lineage\src")

from velocity_ode import load_spatial_dataset
from lineage_trace import DriftMLP, solve_ode_rk4, get_mean_shift_potential_fn

def main():
    real_prep_path = r"D:\001find\project_4_Spatiotemporal_Lineage\data\mosta_preprocessed.h5ad"
    holdout_model_path = r"D:\001find\project_4_Spatiotemporal_Lineage\data\drift_mlp_model_holdout.pt"
    
    if not os.path.exists(real_prep_path):
        print(f"Dataset not found at {real_prep_path}")
        return
        
    print("Loading data...")
    adata = sc.read_h5ad(real_prep_path)
    adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
    adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
    adata_e115 = adata[adata.obs['timepoint'] == 'E11.5'].copy()
    
    coords_e95 = adata_e95.obsm['spatial']
    coords_e105 = adata_e105.obsm['spatial']
    coords_e115 = adata_e115.obsm['spatial']
    
    X_pca_95 = adata_e95.obsm['X_pca']
    state_95 = np.hstack([X_pca_95, coords_e95])
    
    # Load model
    print("Loading model...")
    input_dim = state_95.shape[1]
    model = DriftMLP(input_dim, hidden_dims=[128, 128, 64])
    model.load_state_dict(torch.load(holdout_model_path, map_location=torch.device('cpu')))
    model.eval()
    
    known_spatial = np.vstack([coords_e95, coords_e115])
    
    # Define grid search parameter space
    sigmas = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0]
    etas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    best_ed = float('inf')
    best_params = (None, None)
    
    print("\nRunning grid search...")
    print(f"{'Sigma':<10}{'Eta':<10}{'Energy Distance':<20}")
    print("-" * 40)
    
    for sigma in sigmas:
        for eta in etas:
            # Solve ODE with this potential function
            potential_fn = get_mean_shift_potential_fn(known_spatial, sigma=sigma, eta=eta)
            trajs = solve_ode_rk4(model, state_95, steps=10, potential_grad_fn=potential_fn)
            
            # Predict E10.5 (t=0.5, step=5)
            predicted_spatial = trajs[5][:, -2:]
            
            # Compute ED
            ed = energy_distance(predicted_spatial[:, 0], coords_e105[:, 0]) + \
                 energy_distance(predicted_spatial[:, 1], coords_e105[:, 1])
                 
            print(f"{sigma:<10.1f}{eta:<10.2f}{ed:<20.6f}")
            
            if ed < best_ed:
                best_ed = ed
                best_params = (sigma, eta)
                
    print("-" * 40)
    print(f"Optimal parameters found: Sigma = {best_params[0]}, Eta = {best_params[1]}")
    print(f"Minimum Energy Distance: {best_ed:.6f}")

if __name__ == "__main__":
    main()
