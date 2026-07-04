# Implementation Plan - SpaLineage-OT Pipeline & Scientific Visualizations

This implementation plan details the steps to process the real MOSTA mouse organogenesis dataset, run the spatiotemporal Fused Gromov-Wasserstein OT solver, train the continuous flow matching Neural ODE, and render 5 research-grade publication-ready figures.

## User Review Required

> [!IMPORTANT]
> - **Input Dataset**: The real `mosta.h5ad` contains 54,134 cells. To ensure optimal performance and avoid memory limits on the local hardware, we will run the manifold geodesic solver and Fused Gromov-Wasserstein calculations on a highly representative subset of 4,000 cells (retaining distribution density across E9.5, E10.5, and E11.5).
> - **Velocity Gradient Fallback**: Since the raw `.h5ad` does not contain spliced/unspliced layers for raw RNA velocity calculation, we will calculate **Diffusion Pseudotime (DPT)** using Scanpy/scVelo and compute gradient-based differentiation velocity vectors in the PCA space.
> - **GPU Acceleration**: We will use JAX-accelerated entropic Sinkhorn solvers via `ott-jax` if JAX GPU setup is ready; otherwise, we will use our highly optimized CPU Fused Gromov-Wasserstein solver.

---

## Proposed Changes

### [Data Preprocessing & Manifold Geometry]

#### [MODIFY] [velocity_ode.py](file:///D:/001find/project_4_Spatiotemporal_Lineage/src/velocity_ode.py)
* Update `load_spatial_dataset` to handle the real `mosta.h5ad`:
  * Extract subsets for E9.5, E10.5, and E11.5 developmental stages.
  * Compute PCA (20 dimensions) on the highly variable genes.
  * Compute Diffusion Pseudotime (DPT) using Scanpy to define differentiation root and calculate PCA velocity gradient vectors.
* Implement robust morphological density scaling using KDTree/KDE to simulate physical barriers in the mouse embryo sections.

### [Optimal Transport & Lineage Reconstruction]

#### [MODIFY] [moscot_ot.py](file:///D:/001find/project_4_Spatiotemporal_Lineage/src/moscot_ot.py)
* Refactor `run_spatiotemporal_ot` to run on consecutive stages: E9.5 $\rightarrow$ E10.5 and E10.5 $\rightarrow$ E11.5.
* Standardize cost matrix scales and regularizations to ensure stable convergence of the Polarized Fused Gromov-Wasserstein plan.

#### [MODIFY] [lineage_trace.py](file:///D:/001find/project_4_Spatiotemporal_Lineage/src/lineage_trace.py)
* Update continuous Schrödinger Bridge Flow Matching (SBFM) model to train on joint spatiotemporal states.
* Ensure ODE integrator (RK4) can simulate cell trajectories over developmental stages avoiding high-density morphological barriers using repulsive potential fields.

### [Scientific Visualization Suite]

#### [MODIFY] [visualize_path.py](file:///D:/001find/project_4_Spatiotemporal_Lineage/src/visualize_path.py)
* Implement 5 new functions corresponding to the publication figures:
  1. `plot_delaunay_impedance_mesh`: Overlays Delaunay triangulation on the KDE cell density heatmap.
  2. `plot_euclidean_vs_geodesic_paths`: Compares straight-line path vs. manifold shortest detour path.
  3. `plot_asymmetric_ot_coupling`: Plots the coupling matrix and UMAP velocity alignment.
  4. `plot_continuous_ode_streamlines`: Generates 2D vector streamline flow fields using `streamplot`.
  5. `plot_holdout_validation`: Compares predicted vs. real cell distributions for E10.5.

---

## Verification Plan

### Automated Verification
1. Run the preprocessing and OT mapping:
   `wsl -d Ubuntu /home/jayz/miniconda3/envs/scvi_clear/bin/python /mnt/d/001find/project_4_Spatiotemporal_Lineage/src/moscot_ot.py`
2. Train the continuous flow matching Neural ODE:
   `wsl -d Ubuntu /home/jayz/miniconda3/envs/scvi_clear/bin/python /mnt/d/001find/project_4_Spatiotemporal_Lineage/src/lineage_trace.py`
3. Execute the full visualization suite:
   `wsl -d Ubuntu /home/jayz/miniconda3/envs/scvi_clear/bin/python /mnt/d/001find/project_4_Spatiotemporal_Lineage/src/visualize_path.py`
4. Confirm that the 5 figures (`.png` format) are saved to the artifact directory and verify their layouts.
