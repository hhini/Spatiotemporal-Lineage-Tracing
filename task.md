# Task List - Project 4: SpaLineage-OT Execution & Visualizations

- [x] Download & Verify MOSTA Dataset
  - [x] Download `mosta.h5ad` to `data/`
  - [x] Verify dataset shape, metadata, and timepoints (E9.5, E10.5, E11.5)
- [x] Pipeline Code Refactoring for Real Data
  - [x] Implement DPT & PCA-based differentiation velocity vectors in `src/velocity_ode.py`
  - [x] Adapt `src/moscot_ot.py` to compute stage-to-stage couplings (E9.5 -> E10.5 -> E11.5)
  - [x] Adapt `src/lineage_trace.py` to train continuous Schrödinger Bridge on the real coupling plan
- [x] Implement Publication-Quality Visualization Suite (`src/visualize_path.py`)
  - [x] Figure 1: Delaunay mesh over morphological density (KDE)
  - [x] Figure 2: Euclidean vs. Riemannian geodesic path detour comparison
  - [x] Figure 3: Asymmetric expression coupling heatmap & UMAP velocity overlay
  - [x] Figure 4: Continuous Neural ODE vector field & streamline flow
  - [x] Figure 5: Hold-out validation comparisons (real vs. predicted E10.5)
- [x] Pipeline Execution & Verification
  - [x] Run preprocessing and OT coupling
  - [x] Run flow matching model training
  - [x] Generate all 5 figures and save to artifacts folder
  - [x] Verify image generation and write walkthrough report
