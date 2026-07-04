# Project 4: Data Collection & Preprocessing Checklist

This document details the datasets, databases, accession numbers, and quality control steps required for the Spatiotemporal Lineage Tracing project (SpaLineage-OT).

---

## 1. Primary Datasets
We utilize time-series spatial transcriptomics datasets capturing progressive tissue remodeling (fibrosis/development).

| Dataset Name | Source / Database | Accession Number / URL | Model & Technology | Data Shape / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Bleomycin Lung Fibrosis ST** | NCBI GEO | [GSE198765](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198765) | Mouse Lung at Days 0, 3, 7, 10, 14, 21 (10x Visium) | 12 time-series slices. Includes DAPI H&E. |
| **Mouse Liver Injury Series** | CNGBdb | [CNP0002890](https://db.cngb.org/) | Mouse liver carbon tetrachloride (CCl4) injury series (Stereo-seq) | Sub-micron resolution. High-density time points (Days 0, 1, 3, 7). |
| **Embryonic Development Series** | NCBI GEO | [GSE167890](https://www.ncbi.nlm.nih.gov/geo/) | Time-series mouse gastrulation slices | Validation dataset for developmental dynamics. |

---

## 2. Preprocessing & Quality Control Protocol
1. **Coordinate Alignment (Slices to Common Coordinate Framework - CCF)**:
    - Register consecutive spatial coordinate systems using `PASTE` or `moscot.spatiotemporal.align` to align tissue structures across time.
2. **Spliced/Unspliced Ratio Calculation**:
    - Run `scVelo` / `velocyto` to count spliced and unspliced reads from BAM files (or import precomputed loom matrices) to yield RNA velocity vector fields.
3. **Morphology Impedance Map Reconstruction**:
    - Extract collagen/fibrosis deposition area or DAPI nucleus density from histological images.
    - Compute the local metric tensor $g(x) = I_0 \cdot \exp(\gamma \cdot H(x))$ and construct the Delaunay geodesic distance matrix $\mathbf{D}^{geo}$.
4. **Moscot OT Parameters**:
    - Define expression transport costs based on RNA-velocity warped distance vectors.
    - Set the marginal distributions ($p, q$) using marker-derived growth/death rate estimators.

---

## 3. Benchmark Evaluation Protocols & Metrics

To benchmark `SpaLineage-OT` against **Moscot**, **SpaTrack**, and **WOT**, we implement the following quantitative evaluation protocols:

### 3.1 核心评测实验设计 (Core Evaluation Protocols)
1. **时间点插补预测 (Hold-out Timepoint Reconstruction)**:
    - **协议描述**：在时序数据 $t_1, t_2, t_3$ 中，**抹除中间时间点 $t_2$ 的全部数据**。仅用 $t_1$ 和 $t_3$ 训练最优传输与流匹配（Neural ODE）模型。
    - **执行方式**：利用训练好的流匹配模型 $\mathbf{u}_\theta(\mathbf{x}, t)$ 模拟在 $t=t_2$ 时的空间表达分布。
    - **真值对比**：计算模拟得到的 $t_2$ 表达量密度图与真实的 $t_2$ 空间片区之间的差异。
2. **合成动力学漂移追踪 (Synthetic Drift Tracking)**:
    - **协议描述**：在人工模拟的细胞轨迹数据集上加入物理阻力障碍（Obstacles）。
    - **执行方式**：比较各模型推导出的迁移路径是否会穿过物理障碍。

### 3.2 核心评估指标矩阵 (Metric Matrix)

| 评测维度 (Evaluation Dimension) | 具体评估参数 (Metrics) | 数学定义/物理意义 | 针对的基准模型 (Target Baselines) |
| :--- | :--- | :--- | :--- |
| **插补分布相似度** | **表达重构瓦瑟斯坦距离 (EWD)** | 计算插补出的 $t_2$ 细胞空间分布与真实 $t_2$ 空间分布之间的推土机距离 (Wasserstein-1)。数值越低越优。 | Moscot, WOT |
| **轨迹物理可信度** | **物理屏障越界率 (BCR / OPR)** | 计算推导出的迁移路径与高形态阻抗屏障区域（如致密胶原斑块）的交叉重叠比率。**趋于 0% 最优**。 | Moscot, SpaTrack |
| **方向对齐一致性** | **速度余弦一致性 (VCC / VAS)** | 计算推导的微分轨迹切线 $\frac{d\mathbf{z}}{dt}$ 与独立估算的 RNA 速度向量 $\mathbf{v}$ 之间的余弦相似度。数值越高越优。 | SpaTrack, CellRank |
| **命运转移合理性** | **细胞命运转移一致性 (CTC)** | 统计最优传输耦合矩阵中，符合已知分化动力学规律（如成纤维 $\to$ 肌成纤维）的概率质量比例。 | Moscot, WOT |
