# Project 4: Literature Search and Reference Analysis

This document compiles the core scientific literature underpinning the Spatiotemporal Lineage Tracing Project (SpaLineage-OT).

---

## Paper 1: Moscot Framework for Optimal Transport
- **Title**: *Scalable single-cell optimal transport with moscot*
- **Journal**: *Nature Methods*
- **Publication Date**: 2024
- **DOI/URL**: [10.1101/2023.05.11.540284](https://doi.org/10.1101/2023.05.11.540284)
- **Abstract Summary**: Introduces moscot, a Python library for solving optimal transport problems in single-cell and spatial genomics. It supports multi-omic alignment, spatiotemporal mapping, and lineage reconstruction across time points using Fused Gromov-Wasserstein formulations that scale to millions of cells.
- **Workflow Application**: Provides the underlying computational framework for calculating cell state transitions and mapping ancestral-descendant relationships between successive time points.

---

## Paper 2: SpaTrack modeling
- **Title**: *SpaTrack: reconstructing spatial-temporal cell lineages using optimal transport*
- **Journal**: *Briefings in Bioinformatics*
- **Publication Date**: 2024
- **DOI/URL**: [10.1093/bib/bbae043](https://doi.org/10.1093/bib/bbae043)
- **Abstract Summary**: Introduces SpaTrack, an optimal transport-based tool that integrates both gene expression profiles and spatial distance into transition costs. It reconstructs cell differentiation trajectories at single-cell resolution and models cell fate transitions as a function of time.
- **Workflow Application**: Serves as our primary baseline method for single-cell resolution spatiotemporal trajectory inference.

---

## Paper 3: SOCS (Spatiotemporal OT with Contiguous Structures)
- **Title**: *SOCS: Spatiotemporal Optimal transport with Contiguous Structures for spatial transcriptomics*
- **Journal**: *bioRxiv*
- **Publication Date**: 2025
- **DOI/URL**: [10.1101/2025.xx.xx.xxxxxx](https://doi.org/10.1101/2025.xx.xx.xxxxxx)
- **Abstract Summary**: Proposes a spatial structure-aware optimal transport algorithm that incorporates contiguous anatomical structures into the transport cost, preventing biological incoherence in inferred spatial paths.
- **Workflow Application**: Serves as baseline and justification for replacing Euclidean spatial distances with geodesic distance metrics on morphological manifolds.

---

## Paper 4: Flow Matching for Generative Modeling
- **Title**: *Flow Matching for Generative Modeling*
- **Journal**: *ICLR*
- **Publication Date**: 2023
- **DOI/URL**: [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)
- **Abstract Summary**: Introduces Flow Matching, a simulation-free training method for continuous-time generative models using regression on vector fields. It scales better and trains faster than traditional diffusion models.
- **Workflow Application**: Provides the mathematical formulation for the Schrödinger Bridge Flow Matching (SBFM) decoder, converting discrete OT couplings into continuous physical ODE trajectories.

---

## Paper 5: WOT (Waddington Optimal Transport)
- **Title**: *Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming*
- **Journal**: *Cell*
- **Publication Date**: 2019
- **DOI/URL**: [10.1016/j.cell.2019.01.009](https://doi.org/10.1016/j.cell.2019.01.009)
- **Abstract Summary**: The foundational paper that introduced optimal transport theory to single-cell trajectory inference, showing how time-series snapshots can be stitched together via transport maps.
- **Workflow Application**: Mathematical baseline for temporal coupling and entropy-regularized Sinkhorn algorithms.

---

## Paper 6: MIOFlow (Manifold Interpolating OT Flow)
- **Title**: *Manifold Interpolating Optimal-Transport Flows for Trajectory Inference*
- **Journal**: *arXiv preprint*
- **Publication Date**: 2023
- **DOI/URL**: [https://arxiv.org/abs/2306.02324](https://arxiv.org/abs/2306.02324)
- **Abstract Summary**: Develops a method to learn continuous flows that interpolate between data snapshots by enforcing manifold geodesic distances, ensuring trajectory paths do not cross low-density or invalid states.
- **Workflow Application**: Conceptual basis for our Geodesic Interpolation in Flow Matching, aligning neural trajectories with tissue morphology manifolds.

---

## Paper 7: GENOT (Geometry-Aware Entropic OT Flow)
- **Title**: *GENOT: Geometry-Aware Entropic Optimal Transport for Generative Multi-Omic Mapping*
- **Journal**: *ICML*
- **Publication Date**: 2024
- **DOI/URL**: [https://arxiv.org/abs/2402.12658](https://arxiv.org/abs/2402.12658)
- **Abstract Summary**: Formulates multi-omic mapping and distribution matching by parameterizing the transport map as a geometry-aware neural vector field using entropic OT.
- **Workflow Application**: Informs the integration of unbalanced entropic margins and structural alignment for spatiotemporal dynamics.

