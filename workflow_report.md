# SpaLineage-OT: Spatiotemporal Lineage Tracing Workflow & Scientific Visualization Plan

This report outlines the technical workflow, codebase architecture, and research-grade scientific visualization designs for the **SpaLineage-OT** (Geodesic-Interpolated Unbalanced Schrödinger Bridge Flow Matching) pipeline, executed on the **MOSTA** (Mouse Organogenesis Spatiotemporal Transcriptomic Atlas) dataset.

---

## 1. Mathematical Pipeline & Algorithm Flow

The core methodology is designed as a five-stage sequential pipeline that bridges discrete spatial transcriptomics slices with a continuous-time neural dynamical system.

```mermaid
graph TD
    subgraph Stage 1: Geometry
        D[AnnData: mosta.h5ad] -->|Delaunay Triangulation| M[Riemannian Tissue Manifold]
        M -->|Dijkstra / Shortest Paths| G[Geodesic Cost Matrix D_geo]
    end
    
    subgraph Stage 2: Dynamics
        D -->|Annotation / Pseudotime| V[RNA Velocity Vector Field]
        V -->|Cosine Penalty| A[Asymmetric Cost Matrix C_expr]
    end
    
    subgraph Stage 3: Optimization
        G & A -->|Entropic Polarized Sinkhorn| O[Fused Gromov-Wasserstein Coupling Plan π*]
    end
    
    subgraph Stage 4: Generative Flow
        O -->|Sample Cell Pairs| F[Schrödinger Bridge Flow Matching]
        F -->|Geodesic Tangent Training| U[Neural Drift Field u_θ(x,t)]
    end
    
    subgraph Stage 5: Simulation
        U -->|RK4 Integration + Potential Field| T[Continuous Cell Migration Trajectories]
    end
```

---

## 2. Codebase Architecture & File Responsibilities

The project is structured modularly to separate geometry calculations, optimal transport solving, continuous neural ODE training, and visualization rendering.

```
project_4_Spatiotemporal_Lineage/
├── README.md                      # Project overview, requirements, and execution quickstart
├── literature_search.md           # Curated literature review (GENOT, MIOFlow, BranchSBM, Moscot)
├── data_collection_checklist.md   # Dataset metadata, download FTP paths, and quantitative metrics (EWD, BCR, VCC, CTC)
├── config/
│   └── moscot_config.yaml         # Hyperparameters (OT entropy epsilon, FGW alpha, learning rates, epochs)
├── data/
│   └── mosta.h5ad                 # Processed Mouse Organogenesis dataset (E9.5, E10.5, E11.5)
└── src/
    ├── download_data.py           # Programmatic dataset download script (fallback mirror retrieval)
    ├── velocity_ode.py            # Geometry module (Delaunay triangulation, KDE impedance, Geodesic solver)
    ├── moscot_ot.py               # OT engine (asymmetric VG cost matrix, Fused Gromov-Wasserstein solver)
    ├── lineage_trace.py           # Neural flow matching training and continuous RK4 integration
    └── visualize_path.py          # Visualization suite generating publication-quality figures
```

---

## 3. Research-Grade Scientific Visualizations Plan

To demonstrate the methodological superiority of **SpaLineage-OT**, we have designed 5 advanced, publication-quality scientific figures. Below are the design specifications, biological significance, and visual mapping systems for each figure.

### Figure 1: Delaunay Manifold & Tissue Morphology Impedance Overlay
* **Biological Significance**: Illustrates how the physical tissue environment acts as an impedance barrier (e.g. dense cell cavities, organ boundaries), constraining cellular migration.
* **Visual Mapping**:
  * **Background**: Continuous cell density estimated via Kernel Density Estimation (KDE), plotted as a smooth heatmap with the **`magma`** colormap (representing high-density barriers).
  * **Mesh**: Delaunay triangulation edges overlaid as thin, semi-transparent white lines (`alpha=0.15`) representing the allowed physical migration pathways.
  * **Cell Nodes**: Individual cells plotted as dots colored by their developmental stage (**E9.5, E10.5, E11.5**) with HSL-tailored colors.
* **Implementation Details**:
  * Use `scipy.spatial.Delaunay` for graph construction.
  * Plot the KDE density field using a 2D grid and `matplotlib.pyplot.imshow`.

### Figure 2: Euclidean Straight-line vs. Riemannian Geodesic Detour Comparison
* **Biological Significance**: Demonstrates the mathematical failure of standard Euclidean OT models (which plan trajectories through physical barriers) compared to our Riemannian approach (which plans detour trajectories around barriers).
* **Visual Mapping**:
  * **Barrier Contour**: Dense organ cavity walls marked by a red dotted contour line.
  * **Euclidean Path**: A dashed grey straight line (`style='--'`) crossing straight through the barrier, demonstrating physical impossibility.
  * **Geodesic Path**: A solid glowing cyan curve (`color='#00e5ff'`, `linewidth=2.5`) winding around the high-impedance barrier, showing the calculated manifold path.
* **Implementation Details**:
  * Use `scipy.sparse.csgraph.shortest_path` to retrieve the node-sequence path.
  * Apply Cubic Spline interpolation (`scipy.interpolate.make_interp_spline`) to smooth the discrete node paths into publication-ready curves.

### Figure 3: Asymmetric Expression Coupling Matrix & Velocity Graph Alignment
* **Biological Significance**: Highlights how the asymmetric cost matrix ($C^{VG}_{expr}$) penalizes backwards (reversing) developmental trajectories, matching the physical cell dynamics.
* **Visual Mapping**:
  * **Heatmap**: The solved optimal transport coupling plan $\pi^* \in \mathbb{R}^{n_t \times n_{t+1}}$ plotted as a heatmap using the **`rocket`** colormap, emphasizing high-probability flow paths.
  * **Annotation Scatter**: Adjacent scatter plot of cell states on UMAP coordinate space, overlaid with RNA velocity vector arrows showing the alignment of the transport coupling vectors with the transcription velocity field.
* **Implementation Details**:
  * Draw the coupling matrix heatmap with `seaborn.heatmap`.
  * Plot the velocity field projection using `scanpy.pl.velocity_embedding_grid`.

### Figure 4: Continuous Neural ODE Vector Field & Streamline Flow
* **Biological Significance**: Visualizes the continuous drift vector field $\mathbf{u}_\theta(\mathbf{s}, t)$ learned by the Schrödinger Bridge Flow Matcher, showing the smooth, time-varying fluid-like flow of migrating cells.
* **Visual Mapping**:
  * **Vector Flow**: Background filled with vector streamlines generated using **`matplotlib.pyplot.streamplot`**, where streamline density represents velocity magnitude, colored by the **`viridis`** colormap.
  * **Cell Trajectories**: Overlaid migration trajectories of 20 representative cells from E9.5 to E11.5, plotted as solid black curves with fading alpha and glowing marker points indicating their final positions.
* **Implementation Details**:
  * Evaluate the trained neural model on a uniform 2D spatial grid at time steps $t \in [0, 1]$ to generate the flow grids $(U, V)$ for `streamplot`.

### Figure 5: Hold-out Timepoint Reconstruction Performance (EWD & UMAP Overlap)
* **Biological Significance**: Quantitatively and visually validates our model's interpolation accuracy by comparing our reconstructed E10.5 spatial/expression slice against the real, held-out E10.5 slice.
* **Visual Mapping**:
  * **Left Panel (Spatial Overlay)**: Red points representing the real E10.5 cell positions, and blue points representing the model-predicted E10.5 cell positions, demonstrating spatial registration alignment.
  * **Right Panel (Expression UMAP)**: Contour density plot showing the overlap of the predicted gene expression profiles with the real profiles, demonstrating the high-fidelity reconstruction of intermediate cell states.
* **Implementation Details**:
  * Use UMAP projection coordinates of both real and predicted cell expression states.
  * Plot density contours using `seaborn.kdeplot` with distinct colors for Real vs. Predicted.
