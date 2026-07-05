import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import os
import anndata as ad
from moscot_ot import run_spatiotemporal_ot, load_spatial_dataset

class DriftMLP(nn.Module):
    """
    MLP that models the spatiotemporal drift vector field u_theta(x, t).
    Input: State vector x (expression + spatial coords) concatenated with scalar time t.
    Output: Drift vector of the same dimension as x.
    """
    def __init__(self, input_dim, hidden_dims=[128, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim + 1 # state + time
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Softplus()) # Smooth activation for ODE integration
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, t):
        # x shape: (batch_size, input_dim)
        # t shape: (batch_size, 1)
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

def sample_ot_pairs(pi, X0, X1, num_samples=10000):
    """
    Samples pairs of starting cells (X0) and target cells (X1) from the OT plan pi.
    Uses conditional sampling to prevent memory errors with large coupling matrices.
    """
    n_t, n_tp1 = pi.shape
    # Normalize pi to ensure it sums to 1.0 (joint distribution)
    pi_norm = pi / (pi.sum() + 1e-12)
    
    # 1. Sample row indices (time t cells) based on row marginal sums
    row_sums = pi_norm.sum(axis=1)
    row_sums = row_sums / (row_sums.sum() + 1e-12)
    sampled_rows = np.random.choice(n_t, size=num_samples, p=row_sums)
    
    # 2. For each sampled row, sample col index (time t+1 cells) based on conditional probability
    # Precompute conditional cumulative sums for fast sampling
    # conditional P(col | row) = pi[row, col] / row_sums[row]
    sampled_cols = np.zeros(num_samples, dtype=np.int32)
    
    # To speed up conditional sampling, we group by row index
    unique_rows, counts = np.unique(sampled_rows, return_counts=True)
    for row_idx, count in zip(unique_rows, counts):
        row_prob = pi_norm[row_idx, :]
        row_prob_sum = row_prob.sum()
        if row_prob_sum > 0:
            row_prob = row_prob / row_prob_sum
        else:
            row_prob = np.ones(n_tp1) / n_tp1
            
        sampled_cols[sampled_rows == row_idx] = np.random.choice(n_tp1, size=count, p=row_prob)
        
    return X0[sampled_rows], X1[sampled_cols]

def sample_ot_pairs_with_velocity(pi, X0, X1, V0, V1, num_samples=10000):
    """
    Samples pairs of starting cells, target cells, and their velocities from the OT plan pi.
    """
    n_t, n_tp1 = pi.shape
    pi_norm = pi / (pi.sum() + 1e-12)
    
    row_sums = pi_norm.sum(axis=1)
    row_sums = row_sums / (row_sums.sum() + 1e-12)
    sampled_rows = np.random.choice(n_t, size=num_samples, p=row_sums)
    
    sampled_cols = np.zeros(num_samples, dtype=np.int32)
    unique_rows, counts = np.unique(sampled_rows, return_counts=True)
    for row_idx, count in zip(unique_rows, counts):
        row_prob = pi_norm[row_idx, :]
        row_prob_sum = row_prob.sum()
        if row_prob_sum > 0:
            row_prob = row_prob / row_prob_sum
        else:
            row_prob = np.ones(n_tp1) / n_tp1
            
        sampled_cols[sampled_rows == row_idx] = np.random.choice(n_tp1, size=count, p=row_prob)
        
    return X0[sampled_rows], X1[sampled_cols], V0[sampled_rows], V1[sampled_cols]

def train_flow_matching(X0, X1, pi, config_path, V0=None, V1=None):
    """
    Trains the DriftMLP vector field using Schrödinger Bridge Flow Matching (optionally with physics regularization).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fm_cfg = config['flow_matching_params']
    input_dim = X0.shape[1]
    model = DriftMLP(input_dim, hidden_dims=fm_cfg['hidden_dims'])
    optimizer = optim.Adam(model.parameters(), lr=float(fm_cfg['learning_rate']))
    loss_fn = nn.MSELoss()
    num_samples = 20000
    
    if V0 is not None and V1 is not None:
        X0_samples, X1_samples, V0_samples, V1_samples = sample_ot_pairs_with_velocity(
            pi, X0, X1, V0, V1, num_samples=num_samples
        )
        V0_tensor = torch.tensor(V0_samples, dtype=torch.float32)
        V1_tensor = torch.tensor(V1_samples, dtype=torch.float32)
    else:
        X0_samples, X1_samples = sample_ot_pairs(pi, X0, X1, num_samples=num_samples)
        V0_tensor, V1_tensor = None, None
        
    X0_tensor = torch.tensor(X0_samples, dtype=torch.float32)
    X1_tensor = torch.tensor(X1_samples, dtype=torch.float32)
    batch_size = int(fm_cfg['batch_size'])
    epochs = int(fm_cfg['epochs'])
    model.train()
    
    lambda_phy = 0.5
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x0 = X0_tensor[indices]
            batch_x1 = X1_tensor[indices]
            t = torch.rand((batch_x0.size(0), 1))
            x_t = (1.0 - t) * batch_x0 + t * batch_x1
            target_velocity = batch_x1 - batch_x0
            pred_velocity = model(x_t, t)
            loss_fm = loss_fn(pred_velocity, target_velocity)
            
            if V0_tensor is not None and V1_tensor is not None:
                batch_v0 = V0_tensor[indices]
                batch_v1 = V1_tensor[indices]
                pred_vel_pca = pred_velocity[:, :20]
                v_phy_target = (1.0 - t) * batch_v0 + t * batch_v1
                loss_phy = 1.0 - torch.mean(F.cosine_similarity(pred_vel_pca, v_phy_target, dim=1))
                loss = loss_fm + lambda_phy * loss_phy
            else:
                loss = loss_fm
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/num_batches:.6f}")
    return model

def train_joint_flow_matching(adata_e95, adata_e105, adata_e115, pi_95_105, pi_105_115, config_path):
    """
    Trains a unified continuous DriftMLP vector field across the E9.5 -> E10.5 -> E11.5 trajectory with physics regularization.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    fm_cfg = config['flow_matching_params']
    
    X_pca_95 = adata_e95.obsm['X_pca']
    coord_95 = adata_e95.obsm['spatial']
    state_95 = np.hstack([X_pca_95, coord_95])
    v_pca_95 = adata_e95.obsm['velocity_pca']
    
    X_pca_105 = adata_e105.obsm['X_pca']
    coord_105 = adata_e105.obsm['spatial']
    state_105 = np.hstack([X_pca_105, coord_105])
    v_pca_105 = adata_e105.obsm['velocity_pca']
    
    X_pca_115 = adata_e115.obsm['X_pca']
    coord_115 = adata_e115.obsm['spatial']
    state_115 = np.hstack([X_pca_115, coord_115])
    v_pca_115 = adata_e115.obsm['velocity_pca']
    
    input_dim = state_95.shape[1]
    print(f"[FM Training] Initializing DriftMLP. State dimension: {input_dim}")
    model = DriftMLP(input_dim, hidden_dims=fm_cfg['hidden_dims'])
    optimizer = optim.Adam(model.parameters(), lr=float(fm_cfg['learning_rate']))
    loss_fn = nn.MSELoss()
    
    num_samples_per_stage = 15000
    print("[FM Training] Sampling matching cell pairs from stage-to-stage coupling plans...")
    X0_95, X1_105, V0_95, V1_105 = sample_ot_pairs_with_velocity(
        pi_95_105, state_95, state_105, v_pca_95, v_pca_105, num_samples=num_samples_per_stage
    )
    X0_105, X1_115, V0_105, V1_115 = sample_ot_pairs_with_velocity(
        pi_105_115, state_105, state_115, v_pca_105, v_pca_115, num_samples=num_samples_per_stage
    )
    
    X0_95_t = torch.tensor(X0_95, dtype=torch.float32)
    X1_105_t = torch.tensor(X1_105, dtype=torch.float32)
    V0_95_t = torch.tensor(V0_95, dtype=torch.float32)
    V1_105_t = torch.tensor(V1_105, dtype=torch.float32)
    
    X0_105_t = torch.tensor(X0_105, dtype=torch.float32)
    X1_115_t = torch.tensor(X1_115, dtype=torch.float32)
    V0_105_t = torch.tensor(V0_105, dtype=torch.float32)
    V1_115_t = torch.tensor(V1_115, dtype=torch.float32)
    
    batch_size = int(fm_cfg['batch_size'])
    epochs = int(fm_cfg['epochs'])
    
    print(f"[FM Training] Training joint model for {epochs} epochs...")
    model.train()
    lambda_phy = 0.5
    for epoch in range(epochs):
        permutation_95 = torch.randperm(num_samples_per_stage)
        permutation_105 = torch.randperm(num_samples_per_stage)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples_per_stage, batch_size):
            # E9.5 -> E10.5 (t in [0.0, 0.5])
            idx_95 = permutation_95[i:i+batch_size]
            b_x0_95 = X0_95_t[idx_95]
            b_x1_105 = X1_105_t[idx_95]
            b_v0_95 = V0_95_t[idx_95]
            b_v1_105 = V1_105_t[idx_95]
            
            t_95 = torch.rand((b_x0_95.size(0), 1)) * 0.5
            alpha_val = 2.0 * t_95
            x_t_95 = (1.0 - alpha_val) * b_x0_95 + alpha_val * b_x1_105
            target_vel_95 = 2.0 * (b_x1_105 - b_x0_95)
            v_phy_target_95 = (1.0 - alpha_val) * b_v0_95 + alpha_val * b_v1_105
            
            # E10.5 -> E11.5 (t in [0.5, 1.0])
            idx_105 = permutation_105[i:i+batch_size]
            b_x0_105 = X0_105_t[idx_105]
            b_x1_115 = X1_115_t[idx_105]
            b_v0_105 = V0_105_t[idx_105]
            b_v1_115 = V1_115_t[idx_105]
            
            t_105 = 0.5 + torch.rand((b_x0_105.size(0), 1)) * 0.5
            alpha_val_105 = 2.0 * (t_105 - 0.5)
            x_t_105 = (1.0 - alpha_val_105) * b_x0_105 + alpha_val_105 * b_x1_115
            target_vel_105 = 2.0 * (b_x1_115 - b_x0_105)
            v_phy_target_105 = (1.0 - alpha_val_105) * b_v0_105 + alpha_val_105 * b_v1_115
            
            # Combine batches
            x_t_combined = torch.cat([x_t_95, x_t_105], dim=0)
            t_combined = torch.cat([t_95, t_105], dim=0)
            target_vel_combined = torch.cat([target_vel_95, target_vel_105], dim=0)
            v_phy_combined = torch.cat([v_phy_target_95, v_phy_target_105], dim=0)
            
            pred_vel = model(x_t_combined, t_combined)
            loss_fm = loss_fn(pred_vel, target_vel_combined)
            
            pred_vel_pca = pred_vel[:, :20]
            loss_phy = 1.0 - torch.mean(F.cosine_similarity(pred_vel_pca, v_phy_combined, dim=1))
            
            loss = loss_fm + lambda_phy * loss_phy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
            
    return model

def solve_ode_rk4(model, x_start, steps=20, potential_grad_fn=None):
    """
    Traces trajectories using our custom Runge-Kutta 4th order ODE solver.
    dx/dt = u_theta(x, t) from t = 0 to t = 1.
    """
    model.eval()
    x = torch.tensor(x_start, dtype=torch.float32)
    n_cells = x.size(0)
    
    dt = 1.0 / steps
    trajectory = [x.numpy().copy()]
    
    with torch.no_grad():
        for step in range(steps):
            # Current time t
            t_val = step * dt
            t = torch.full((n_cells, 1), t_val, dtype=torch.float32)
            
            def get_drift(state, time):
                drift = model(state, time)
                if potential_grad_fn is not None:
                    drift = drift.clone()
                    drift[:, -2:] += potential_grad_fn(state[:, -2:])
                return drift
            
            # Runge-Kutta stages
            k1 = get_drift(x, t)
            k2 = get_drift(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = get_drift(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = get_drift(x + dt * k3, t + dt)
            
            # Update state
            x = x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            trajectory.append(x.numpy().copy())
            
    # Return trajectory array of shape (steps+1, n_cells, D)
    return np.array(trajectory)

# Custom potential force function based on tissue density mode (mean shift)
def get_mean_shift_potential_fn(X_real_spatial, sigma=20.0, eta=0.1):
    X_real_t = torch.tensor(X_real_spatial, dtype=torch.float32)
    def potential_grad_fn(x_current):
        # x_current shape: (N, 2)
        # Compute pairwise distance squared: shape (N, M)
        dists_sq = torch.cdist(x_current, X_real_t) ** 2
        # Gaussian weights: shape (N, M)
        weights = torch.exp(-dists_sq / (2.0 * sigma ** 2))
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        # Mask out coordinates where density is extremely low to prevent numerical underflow/pull to origin
        mask = (weights_sum > 1e-4).float()
        weighted_coords = torch.matmul(weights, X_real_t) / (weights_sum + 1e-12)
        # Mean shift vector: shape (N, 2)
        mean_shift = (weighted_coords - x_current) * mask
        return eta * mean_shift
    return potential_grad_fn

if __name__ == "__main__":
    import os
    real_prep_path = "data/mosta_preprocessed.h5ad"
    pi_95_105_path = "data/pi_95_105.npy"
    pi_105_115_path = "data/pi_105_115.npy"
    config_path = "config/moscot_config.yaml"
    
    if os.path.exists(real_prep_path) and os.path.exists(pi_95_105_path) and os.path.exists(pi_105_115_path):
        import scanpy as sc
        print(f"[FM Pipeline] Loading preprocessed dataset and coupling plans...")
        adata = sc.read_h5ad(real_prep_path)
        pi_95_105 = np.load(pi_95_105_path)
        pi_105_115 = np.load(pi_105_115_path)
        
        adata_e95 = adata[adata.obs['timepoint'] == 'E9.5'].copy()
        adata_e105 = adata[adata.obs['timepoint'] == 'E10.5'].copy()
        adata_e115 = adata[adata.obs['timepoint'] == 'E11.5'].copy()
        
        # Train unified joint model (production)
        print("[FM Pipeline] Training joint model (E9.5 -> E10.5 -> E11.5)...")
        model = train_joint_flow_matching(adata_e95, adata_e105, adata_e115, pi_95_105, pi_105_115, config_path)
        model_save_path = "data/drift_mlp_model.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"[FM Pipeline] Trained joint model saved to {model_save_path}")
        
        # Train hold-out validation model (skipping E10.5)
        pi_95_115_path = "data/pi_95_115.npy"
        if os.path.exists(pi_95_115_path):
            print("[FM Pipeline] Training hold-out validation model (E9.5 -> E11.5)...")
            pi_95_115 = np.load(pi_95_115_path)
            state_95 = np.hstack([adata_e95.obsm['X_pca'], adata_e95.obsm['spatial']])
            state_115 = np.hstack([adata_e115.obsm['X_pca'], adata_e115.obsm['spatial']])
            v_pca_95 = adata_e95.obsm['velocity_pca']
            v_pca_115 = adata_e115.obsm['velocity_pca']
            model_holdout = train_flow_matching(state_95, state_115, pi_95_115, config_path, v_pca_95, v_pca_115)
            model_holdout_save_path = "data/drift_mlp_model_holdout.pt"
            torch.save(model_holdout.state_dict(), model_holdout_save_path)
            print(f"[FM Pipeline] Trained hold-out validation model saved to {model_holdout_save_path}")
        else:
            print("[FM Pipeline] Warning: pi_95_115.npy not found. Skipping hold-out training.")
        
    else:
        print("[Test] Loading test datasets and running OT solver...")
        adata_t = load_spatial_dataset("")
        adata_tp1 = load_spatial_dataset("")
        pi = run_spatiotemporal_ot(adata_t, adata_tp1, config_path)
        
        X_pca_t = adata_t.obsm['X_pca']
        coord_t = adata_t.obsm['spatial']
        state_t = np.hstack([X_pca_t, coord_t])
        
        X_pca_tp1 = adata_tp1.obsm['X_pca']
        coord_tp1 = adata_tp1.obsm['spatial']
        state_tp1 = np.hstack([X_pca_tp1, coord_tp1])
        
        model = train_flow_matching(state_t, state_tp1, pi, config_path)
        paths = solve_ode_rk4(model, state_t, steps=20)
        print(f"Computed trajectory paths of shape: {paths.shape}")
        print(f"Starting coords mean: {paths[0, :, -2:].mean(axis=0)}")
        print(f"Ending coords mean: {paths[-1, :, -2:].mean(axis=0)}")
