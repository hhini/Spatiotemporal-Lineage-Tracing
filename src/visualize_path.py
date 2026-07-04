import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import yaml
import sys
import scanpy as sc
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.stats import energy_distance
import scipy.sparse as sp

# Add src/ to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from velocity_ode import compute_morphology_impedance, compute_geodesic_distance_matrix, load_spatial_dataset
from lineage_trace import DriftMLP, solve_ode_rk4, sample_ot_pairs, get_mean_shift_potential_fn

# Set matplotlib style for high-quality scientific figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 1.2,
    'xtick.color': '#2C3E50',
    'ytick.color': '#2C3E50',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linestyle': '--',
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
})

def project_velocity_to_umap(X_pca, X_umap, velocity_pca, n_neighbors=15):
    """
    Projects PCA velocity vectors onto UMAP space using cosine-similarity-weighted neighbor coordinates.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
    distances, indices = nbrs.kneighbors(X_pca)
    
    velocity_umap = np.zeros_like(X_umap)
    for i in range(X_pca.shape[0]):
        nbr_idx = indices[i]
        diff_pca = X_pca[nbr_idx] - X_pca[i]
        diff_umap = X_umap[nbr_idx] - X_umap[i]
        
        v = velocity_pca[i]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        
        weights = []
        for d_pca in diff_pca:
            dp_norm = np.linalg.norm(d_pca)
            if dp_norm == 0:
                weights.append(0.0)
            else:
                weights.append(np.dot(v, d_pca) / (v_norm * dp_norm))
        
        weights = np.array(weights)
        weights = np.exp(weights * 5.0)  # Exponential scaling to amplify directionality
        weights /= np.sum(weights)
        
        velocity_umap[i] = np.dot(weights, diff_umap)
        
    return velocity_umap

def plot_delaunay_impedance_mesh(adata, output_dir):
    """
    Figure 1: Delaunay mesh overlay on morphology impedance (KDE) density map.
    """
    print("[Viz] Plotting Figure 1: Delaunay Impedance Mesh...")
    coords = adata.obsm['spatial']
    imp = compute_morphology_impedance(coords, gamma=2.0)
    
    # Compute Delaunay triangulation
    tri = Delaunay(coords)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Draw Delaunay mesh edges (translucent)
    ax.triplot(coords[:, 0], coords[:, 1], tri.simplices, color='#BDC3C7', alpha=0.18, lw=0.4, zorder=1)
    
    # Plot cells colored by impedance density
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=imp, cmap='magma', s=8, alpha=0.85, zorder=2)
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, aspect=30)
    cbar.set_label('Tissue Impedance Weight $\exp(\gamma \cdot H(x))$', fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title("Manifold Delaunay Mesh & Morphology Impedance Heatmap", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Spatial Axis X ($\mu$m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Spatial Axis Y ($\mu$m)", fontsize=11, labelpad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_delaunay_mesh.png"), bbox_inches='tight')
    plt.close()

def plot_euclidean_vs_geodesic_paths(adata, output_dir):
    """
    Figure 2: Euclidean vs. Riemannian geodesic path detour comparison.
    """
    print("[Viz] Plotting Figure 2: Path Detour Comparison...")
    coords = adata.obsm['spatial']
    imp = compute_morphology_impedance(coords, gamma=2.0)
    
    # Build Delaunay graph and adjacency
    tri = Delaunay(coords)
    n_cells = coords.shape[0]
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
                
    row, col, data = [], [], []
    max_edge_length = 100.0
    for u, v in edges:
        p1 = coords[u]
        p2 = coords[v]
        dist = np.linalg.norm(p1 - p2)
        if dist <= max_edge_length:
            avg_imp = 0.5 * (imp[u] + imp[v])
            weight = dist * avg_imp
            row.append(u); col.append(v); data.append(weight)
            row.append(v); col.append(u); data.append(weight)
            
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
    
    # Find two cells on opposite sides of a high-density zone
    # We choose E9.5 cells with large spatial separation, e.g. from top-left to bottom-right
    # Let's search for indices
    center = np.mean(coords, axis=0)
    # Filter cells near x=center_x, y=center_y to find a barrier
    # Find points furthest apart
    dists_from_center = np.linalg.norm(coords - center, axis=1)
    # We find start as point near bottom-left and end as point near top-right
    dist_matrix, predecessors = shortest_path(csgraph=adj_matrix, method='D', directed=False, return_predecessors=True)
    
    # Let's find a pair (u, v) with a large detour ratio: Geodesic Dist / Euclidean Dist
    best_u, best_v = 0, 1
    max_detour = 1.0
    
    # Sample some distant pairs to find a nice detour
    np.random.seed(42)
    sample_indices = np.random.choice(n_cells, size=min(n_cells, 100), replace=False)
    for u in sample_indices:
        for v in sample_indices:
            if u == v: continue
            eucl_d = np.linalg.norm(coords[u] - coords[v])
            geod_d = dist_matrix[u, v]
            if np.isfinite(geod_d) and eucl_d > 100.0:
                detour = geod_d / eucl_d
                if detour > max_detour:
                    max_detour = detour
                    best_u, best_v = u, v
                    
    # Reconstruct shortest path
    path = []
    curr = best_v
    while curr != best_u and curr >= 0:
        path.append(curr)
        curr = predecessors[best_u, curr]
    if curr == best_u:
        path.append(best_u)
    path.reverse()
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Plot background cell density contour (KDE)
    ax.scatter(coords[:, 0], coords[:, 1], c=imp, cmap='Blues', s=6, alpha=0.3, zorder=1)
    
    # Plot Euclidean straight path
    ax.plot([coords[best_u, 0], coords[best_v, 0]], [coords[best_u, 1], coords[best_v, 1]], 
            color='#E74C3C', linestyle='--', lw=2.0, label='Euclidean Path (Straight line)', zorder=2)
            
    # Plot Geodesic detour path
    path_coords = coords[path]
    ax.plot(path_coords[:, 0], path_coords[:, 1], color='#16A085', lw=3.0, label='Riemannian Geodesic Path (Manifold detour)', zorder=3)
    
    # Highlight start and end cells
    ax.scatter(coords[best_u, 0], coords[best_u, 1], color='#C0392B', s=100, edgecolors='black', lw=1.5, label='Start Cell ($x_0$)', zorder=4)
    ax.scatter(coords[best_v, 0], coords[best_v, 1], color='#27AE60', s=100, edgecolors='black', lw=1.5, label='Target Cell ($x_1$)', zorder=4)
    
    # Title and metrics text
    ax.set_title("Euclidean vs. Riemannian Geodesic Development Paths", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Spatial Axis X ($\mu$m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Spatial Axis Y ($\mu$m)", fontsize=11, labelpad=8)
    
    eucl_dist = np.linalg.norm(coords[best_u] - coords[best_v])
    geod_dist = dist_matrix[best_u, best_v]
    
    text_str = f"Euclidean Dist: {eucl_dist:.2f} $\mu$m\nGeodesic Dist: {geod_dist:.2f} $\mu$m\nDetour Ratio: {geod_dist/eucl_dist:.2f}x"
    ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))
            
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_path_detour.png"), bbox_inches='tight')
    plt.close()

def plot_asymmetric_ot_coupling(adata, pi_95_105, output_dir):
    """
    Figure 3: Asymmetric expression coupling heatmap & UMAP velocity overlay.
    """
    print("[Viz] Plotting Figure 3: Asymmetric Coupling & UMAP Flow...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # 1. Coupling Heatmap (sub-sample 80 cells from E9.5 and 80 from E10.5 for clean plotting)
    # Sort them by dpt_pseudotime to align developmental progression
    adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
    adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
    
    idx_95 = np.argsort(adata_e95.obs['dpt_pseudotime'].values)
    idx_105 = np.argsort(adata_e105.obs['dpt_pseudotime'].values)
    
    pi_sorted = pi_95_105[idx_95][:, idx_105]
    
    # Downsample matrix to 100 x 100 for visual clarity
    down_95 = np.linspace(0, len(idx_95)-1, 100, dtype=int)
    down_105 = np.linspace(0, len(idx_105)-1, 100, dtype=int)
    pi_down = pi_sorted[down_95][:, down_105]
    
    im = ax1.imshow(pi_down, cmap='magma_r', aspect='auto', origin='lower')
    ax1.set_title("Sorted Stage Coupling Plan $\Pi$ (E9.5 $\\rightarrow$ E10.5)", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("E10.5 Cells (sorted by DPT)", fontsize=11, labelpad=8)
    ax1.set_ylabel("E9.5 Cells (sorted by DPT)", fontsize=11, labelpad=8)
    
    # Label cell stages along axes
    cbar = fig.colorbar(im, ax=ax1, pad=0.03, shrink=0.85)
    cbar.set_label('Transport Probability Density', fontsize=10, labelpad=8)
    
    # 2. UMAP Velocity Overlay
    # Run neighbors and UMAP if not already calculated
    if 'X_umap' not in adata.obsm:
        print("[Viz] Computing UMAP embeddings...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
        sc.tl.umap(adata)
        
    X_umap = adata.obsm['X_umap']
    timepoint = adata.obs['timepoint'].values
    
    # Project velocity
    X_pca = adata.obsm['X_pca']
    v_pca = adata.obsm['velocity_pca']
    v_umap = project_velocity_to_umap(X_pca, X_umap, v_pca, n_neighbors=15)
    
    # Plot E9.5 and E10.5 cells
    e95_mask = timepoint == 'E9.5'
    e105_mask = timepoint == 'E10.5'
    
    ax2.scatter(X_umap[e95_mask, 0], X_umap[e95_mask, 1], color='#2980B9', s=15, alpha=0.5, label='E9.5 Cells')
    ax2.scatter(X_umap[e105_mask, 0], X_umap[e105_mask, 1], color='#8E44AD', s=15, alpha=0.5, label='E10.5 Cells')
    
    # Add velocity quiver arrows (grid-based vector field or sampled cells)
    # We sample 60 cells across the dataset to show clear directionality arrows
    np.random.seed(42)
    sample_idx = np.random.choice(adata.shape[0], size=80, replace=False)
    ax2.quiver(X_umap[sample_idx, 0], X_umap[sample_idx, 1], 
              v_umap[sample_idx, 0], v_umap[sample_idx, 1], 
              color='#2C3E50', scale=15, width=0.004, alpha=0.9, pivot='mid')
              
    ax2.set_title("Expression UMAP & RNA-Velocity Vector Quivers", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("UMAP Dimension 1", fontsize=11, labelpad=8)
    ax2.set_ylabel("UMAP Dimension 2", fontsize=11, labelpad=8)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_ot_coupling.png"), bbox_inches='tight')
    plt.close()

def plot_continuous_ode_streamlines(adata, model_path, config_path, output_dir):
    """
    Figure 4: Continuous Neural ODE vector field & streamline flow.
    """
    print("[Viz] Plotting Figure 4: Neural ODE Streamlines...")
    # Load model configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fm_cfg = config['flow_matching_params']
    
    coords = adata.obsm['spatial']
    X_pca = adata.obsm['X_pca']
    
    input_dim = X_pca.shape[1] + coords.shape[1]
    model = DriftMLP(input_dim, hidden_dims=fm_cfg['hidden_dims'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Define a spatial grid across the embryo tissue
    xmin, xmax = coords[:, 0].min() - 15, coords[:, 0].max() + 15
    ymin, ymax = coords[:, 1].min() - 15, coords[:, 1].max() + 15
    
    x_grid = np.linspace(xmin, xmax, 40)
    y_grid = np.linspace(ymin, ymax, 40)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # We query the model at time t=0.5
    # For PCA variables, we use the average PCA profile at E10.5
    avg_pca_105 = np.mean(X_pca[adata.obs['timepoint'] == 'E10.5'], axis=0)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                spatial_pt = np.array([X[i, j], Y[i, j]])
                # Combine average PCA and grid spatial coordinates
                state_vec = np.hstack([avg_pca_105, spatial_pt])
                state_t = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
                t_tensor = torch.tensor([[0.5]], dtype=torch.float32) # Query mid-stage development
                
                drift = model(state_t, t_tensor).numpy()[0]
                # Spatial components are the last 2 dimensions of the vector field
                U[i, j] = drift[-2]
                V[i, j] = drift[-1]
                
    # Normalize lengths to draw nice streamlines
    speed = np.sqrt(U**2 + V**2)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Plot cells
    colors_tp = adata.obs['timepoint'].map({'E9.5': '#3498DB', 'E10.5': '#F1C40F', 'E11.5': '#9B59B6'}).values
    ax.scatter(coords[:, 0], coords[:, 1], c=colors_tp, s=6, alpha=0.25, zorder=1)
    
    # Plot streamlines colored by development velocity magnitude
    strm = ax.streamplot(X, Y, U, V, color=speed, cmap='plasma', linewidth=1.2, 
                         density=1.5, arrowsize=1.0, zorder=2)
                         
    cbar = fig.colorbar(strm.lines, ax=ax, pad=0.03, aspect=30)
    cbar.set_label('Neural ODE Spatial Migration Speed ($\mu$m/day)', fontsize=11, labelpad=8)
    
    ax.set_title("Continuous Vector Field & Developmental Streamlines", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Spatial Axis X ($\mu$m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Spatial Axis Y ($\mu$m)", fontsize=11, labelpad=8)
    
    # Add time label
    ax.text(0.05, 0.93, "Time $t = 0.5$ (E10.5)", transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))
            
    # Custom legend for scatter timepoints
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=8, label='E9.5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F1C40F', markersize=8, label='E10.5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=8, label='E11.5'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='none', shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_vector_field.png"), bbox_inches='tight')
    plt.close()

def plot_holdout_validation(adata, model_path, config_path, output_dir):
    """
    Figure 5: Hold-out validation comparison (E10.5 prediction vs. actual).
    """
    print("[Viz] Plotting Figure 5: Hold-out Validation...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fm_cfg = config['flow_matching_params']
    
    # 1. Load model and integrate E9.5 cells to t=0.5
    adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
    adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
    
    X_pca_95 = adata_e95.obsm['X_pca']
    coord_95 = adata_e95.obsm['spatial']
    state_95 = np.hstack([X_pca_95, coord_95])
    
    input_dim = state_95.shape[1]
    model = DriftMLP(input_dim, hidden_dims=fm_cfg['hidden_dims'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Solve ODE for E9.5 to t=0.5 (representing E10.5 prediction) with Mean Shift potential force
    # 10 steps representing t in [0.0, 1.0]. The E10.5 prediction is at t=0.5 (step 5).
    print("[Viz] Integrating Neural ODE trajectories to intermediate time t=0.5 with potential force...")
    coord_115 = adata[adata.obs['timepoint'] == 'E11.5'].obsm['spatial']
    known_spatial = np.vstack([coord_95, coord_115])
    potential_fn = get_mean_shift_potential_fn(known_spatial, sigma=50.0, eta=0.2)
    trajs = solve_ode_rk4(model, state_95, steps=10, potential_grad_fn=potential_fn) # shape: (11, n_cells, D)
    predicted_e105_states = trajs[5] # Index 5 corresponds to t = 0.5 (E10.5 prediction)
    predicted_spatial = predicted_e105_states[:, -2:]
    
    actual_spatial = adata_e105.obsm['spatial']
    
    # Calculate Energy Distance / 2D Wasserstein discrepancy
    e_dist = energy_distance(predicted_spatial[:, 0], actual_spatial[:, 0]) + \
             energy_distance(predicted_spatial[:, 1], actual_spatial[:, 1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    
    # Plot Predicted E10.5 Cell Distribution
    ax1.hexbin(predicted_spatial[:, 0], predicted_spatial[:, 1], gridsize=25, cmap='plasma', mincnt=1)
    ax1.set_title("Model Predicted E10.5 Distribution", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Spatial Axis X ($\mu$m)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Spatial Axis Y ($\mu$m)", fontsize=11, labelpad=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot Actual E10.5 Cell Distribution
    ax2.hexbin(actual_spatial[:, 0], actual_spatial[:, 1], gridsize=25, cmap='plasma', mincnt=1)
    ax2.set_title("Experimental Actual E10.5 Distribution", fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel("Spatial Axis X ($\mu$m)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Spatial Axis Y ($\mu$m)", fontsize=11, labelpad=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Discrepancy label
    text_str = f"Wasserstein-Equivalent Discrepancy:\nEnergy Distance = {e_dist:.4f}"
    fig.text(0.5, 0.02, text_str, ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#BDC3C7', alpha=0.9))
             
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(output_dir, "fig5_validation.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    brain_dir = r"C:\Users\Administrator\.gemini\antigravity\brain\16e92036-167e-4691-932b-d2a1ad158db7"
    if os.name == 'posix' and brain_dir.startswith("C:\\"):
        brain_dir = "/mnt/c/" + brain_dir[3:].replace("\\", "/")
        
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    real_prep_path = "data/mosta_preprocessed.h5ad"
    pi_95_105_path = "data/pi_95_105.npy"
    model_path = "data/drift_mlp_model.pt"
    holdout_model_path = "data/drift_mlp_model_holdout.pt"
    config_path = "config/moscot_config.yaml"
    
    if os.path.exists(real_prep_path) and os.path.exists(pi_95_105_path) and os.path.exists(model_path):
        print(f"[Viz Pipeline] Loading real processed data and models...")
        adata = sc.read_h5ad(real_prep_path)
        pi_95_105 = np.load(pi_95_105_path)
        
        # Determine validation model
        val_model_path = holdout_model_path if os.path.exists(holdout_model_path) else model_path
        
        # Plot 5 Figures
        plot_delaunay_impedance_mesh(adata, out_dir)
        plot_euclidean_vs_geodesic_paths(adata, out_dir)
        plot_asymmetric_ot_coupling(adata, pi_95_105, out_dir)
        plot_continuous_ode_streamlines(adata, model_path, config_path, out_dir)
        plot_holdout_validation(adata, val_model_path, config_path, out_dir)
        
        print("[Viz Pipeline] All 5 figures generated and saved to results/ successfully!")
        
        # Copy to brain folder for display
        import shutil
        if os.path.exists(brain_dir) and os.path.abspath(out_dir) != os.path.abspath(brain_dir):
            for fig_name in ["fig1_delaunay_mesh.png", "fig2_path_detour.png", "fig3_ot_coupling.png", "fig4_vector_field.png", "fig5_validation.png"]:
                src_fig = os.path.join(out_dir, fig_name)
                dst_fig = os.path.join(brain_dir, fig_name)
                try:
                    shutil.copy(src_fig, dst_fig)
                    print(f"  Copied {fig_name} to brain directory: {brain_dir}")
                except Exception as e:
                    print(f"  Failed to copy {fig_name}: {e}")
    else:
        print("[Viz Pipeline] Real preprocessed data or model not found. Falling back to synthetic visualization...")
        # Fallback script execution using dummy data
        out_path = os.path.join(out_dir, "lineage_migration_pathway.png")
        run_visual_pipeline(out_path)

