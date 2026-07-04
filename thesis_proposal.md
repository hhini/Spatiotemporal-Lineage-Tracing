# 基于时空最优传输与动力学常微分方程的空间谱系追踪算法研究

## 1. 课题背景与立项依据

### 1.1 科学背景与病理意义
在器官发育、组织再生以及诸如纤维化（Fibrosis）和肿瘤转移等病理过程中，细胞命运决定与其空间微环境（Microenvironment）紧密相关。以肺纤维化（Lung Fibrosis）为例，常驻成纤维细胞（Resident Fibroblasts）在炎症及细胞因子刺激下，会跨越不同的组织分区向激活态的肌成纤维细胞（Myofibroblasts）分化，并物理迁移至肺泡间隔或纤维化核心区，分泌大量细胞外基质（ECM），导致肺结构毁损。

理解这种复杂的细胞分化与空间迁移动力学，对于揭示病理机制、发现早期干预靶点至关重要。

### 1.2 现有技术的瓶颈
虽然单细胞转录组测序（scRNA-seq）与空间转录组学（ST）取得了飞速发展，但重建连续的时空发育和迁移谱系仍面临重大技术瓶颈：
1. **时间快照限制（Static Snapshots）**：测序和成像实验具有破坏性，我们只能获得离散时间点（如 Day 0, 3, 7, 14）的细胞“静态快照”，无法直接观测同一细胞的连续发育轨迹。
2. **时空维度解耦（Spatiotemporal Decoupling）**：传统轨迹推断算法（如 Monocle, CellRank）仅在转录组表征空间中推算“伪时间（Pseudotime）”，完全丢失了细胞在组织中的物理空间运动路径；而传统的空间对齐算法（如 PASTE）仅关注空间形状配准，忽略了分化过程的定向驱动力。
3. **数理模型缺乏物理屏障感知**：现有的时空最优传输模型（如 Moscot [1]）计算空间转移距离时均采用简单的欧氏距离。在生物组织中，细胞无法穿过高密度的形态阻抗屏障（如致密的纤维化空腔、骨屏障），这导致推导出的物理迁移轨迹常常违背基本的解剖学常识。

因此，亟需发展一种**融合 RNA 动力学矢量场与形态阻抗流形测地线约束的连续时空最优传输算法**，以实现高精度的时空谱系追踪。

---

## 2. 国内外研究现状与文献综述

时空谱系追踪与轨迹推断目前是系统生物学和生物信息学的核心前沿。其技术演进路线可分为以下三个阶段：

### 2.1 纯转录空间轨迹推断（无空间信息）
早期的研究主要基于解离的单细胞数据。**WOT (Waddington Optimal Transport)** [2] 首次将最优传输理论引入单细胞领域，通过建立相邻时间点细胞群体的输运图（Transport Map），重建细胞的发育流向。随后，**scVelo** [3] 利用 spliced/unspliced RNA 转录速率建立常微分方程（ODE），预测细胞的即时状态变化矢量（RNA Velocity）。然而，这些方法只能预测细胞在“状态空间”中的转移，无法提供细胞在物理空间中的三维运动路径。

### 2.2 空间几何对齐与多时间点映射
随着空间转录组的兴起，**PASTE** [4] 引入融合格罗莫夫-瓦瑟斯坦（Fused Gromov-Wasserstein, FGW）算法，实现了相邻切片间的刚性与非刚性物理对齐。近期的 **Moscot** [1] 框架将单细胞和空间多组学数据整合，利用极化 Sinkhorn 算法求解大规模最优传输问题，支持跨时间点的空间动力学映射。然而，Moscot 的 `spatiotemporal` 模块在对齐时，仅利用了静态的转录相似度与欧氏空间距离，忽略了 RNA 速度所包含的指向性分化动能，且容易得出穿透物理屏障的“假阳性迁移路径”。

### 2.3 融合多模态与结构约束的前沿算法
最新发表的 **SpaTrack** [5] 尝试利用最优传输直接计算单细胞分辨率下的空间分化轨迹，并结合转录因子（TF）驱动力进行因果推断。同时，**SOCS (Spatiotemporal Optimal transport with Contiguous Structures)** [6] 在最优传输中引入了局部空间邻近的连续性结构约束，防止空间结构解离。
然而，**如何系统性地将 RNA 动力学速率（微分方程）与空间最优传输（积分约束）进行数理上的无缝融合，并建立感知物理阻抗的非欧测地线输运路径**，目前在国际上仍属空白。这正是本课题的核心切入点。

---

## 3. 核心创新点与数理推导

本项目提出了一套全新的时空轨迹重建数理框架 **`SpaLineage-OT`**，其数学创新体现在以下三个核心方向：

### 3.1 创新点一：RNA 速度场引导的非对称动力学输运代价（VG-OT）
传统的最优传输在转录隐空间 $Z$ 中使用对称的表达成本：
$$C_{expr}(i, j) = 1 - \cos(\mathbf{z}_i, \mathbf{z}_j)$$
为了引入分化方向性约束，我们利用 `scVelo` 估计出的潜在表达速度向量 $\mathbf{v}_i \in \mathbb{R}^D$。对于时间 $t$ 处的细胞 $i$ 与时间 $t+1$ 处的候选子代细胞 $j$，其转移位移向量为 $\mathbf{d}_{ij} = \mathbf{z}_j - \mathbf{z}_i$。

我们构造**速度场引导的非对称传输代价函数 (Velocity-Guided Cost)**：
$$C_{expr}^{VG}(i, j) = 1 - \cos(\mathbf{z}_i, \mathbf{z}_j) + \beta \cdot D_{drift}(\mathbf{d}_{ij}, \mathbf{v}_i)$$
其中，漂移惩罚项定义为位移向量与速度向量的负向夹角余弦：
$$D_{drift}(\mathbf{d}_{ij}, \mathbf{v}_i) = 1 - \frac{\langle \mathbf{d}_{ij}, \mathbf{v}_i \rangle}{\|\mathbf{d}_{ij}\|_2 \|\mathbf{v}_i\|_2 + \eta}$$
*   **数理逻辑**：若细胞分化位移 $\mathbf{d}_{ij}$ 的方向与 RNA 速度向量 $\mathbf{v}_i$ 指向一致，则余弦接近 1，$D_{drift} \to 0$，传输成本极低；若位移方向逆速度矢量而行，则面临高额代价惩罚 $\beta$。这打破了传统最优传输的对称性，强制了时间流向和发育层级的单向性。

### 3.2 创新点二：形态学阻抗流形测地线格罗莫夫-瓦瑟斯坦（G-FGW）
传统空间 GW 距离基于欧氏距离：
$$d_{Euc}(x_a, x_b) = \|x_a - x_b\|_2$$
在实体组织中，细胞迁移受细胞外基质（如胶原纤维）沉积或细胞高密度区的影响。我们将组织空间建模为黎曼流形 $(\mathcal{M}, g)$，其局部度规张量 $g(x)$ 定义为：
$$g(x) = I_0 \cdot \exp(\gamma \cdot H(x))$$
其中 $H(x)$ 为位置 $x$ 处的免疫荧光形态学灰度（如胶原蛋白 Collagen 染色强度或 DAPI 细胞核密度），$\gamma$ 为阻抗敏感系数。

我们构建空间 Delaunay 图，边的权重依据局部阻抗重新加权：
$$w_{ab} = d_{Euc}(x_a, x_b) \cdot \int_{0}^{1} g((1-\tau)x_a + \tau x_b) d\tau$$
利用 Floyd-Warshall 或 Dijkstra 算法计算全局对偶最短路径，得到物理感知**测地线距离 (Geodesic Distance)** $d_{geo}(x_a, x_b)$。将 Gromov-Wasserstein 的输入距离矩阵替换为测地线距离矩阵 $\mathbf{D}^{geo}$：
$$\text{GW}_{geo}(\mu_t, \mu_{t+1}, \pi) = \sum_{a,b,a',b'} |d_{geo}(x_a, x_b) - d_{geo}(y_{a'}, y_{b'})|^2 \pi_{a,a'} \pi_{b,b'}$$
*   **数理逻辑**：在阻抗极高（如重度纤维化区）的区域，测地线距离远大于几何距离，从而有效阻止了输运映射 $\pi^*$ 穿透这些物理屏障。

### 3.3 创新点三：基于流形测地路径插值与非平衡约束的连续薛定谔桥流匹配 (GI-USBFM)

为了将离散的输运图 $\pi^*$ 转化为连续时间的动力学轨迹，并内生性地感应空间物理屏障与细胞质量消长，本项目提出**基于流形测地路径插值与非平衡约束的薛定谔桥流匹配 (Geodesic-Interpolated Unbalanced Schrödinger Bridge Flow Matching, GI-USBFM)**：

1. **引入非平衡最优传输 (Unbalanced OT)**：
   细胞发育伴随分裂与凋亡。我们在第一阶段计算中采用非平衡格罗莫夫-瓦瑟斯坦 (Unbalanced Fused Gromov-Wasserstein, U-FGW) 约束，允许边缘分布与真实观测存在偏差，以刻画群体质量的非保守性变化：
   $$\min_{\pi \ge 0} \langle \pi, C^{VG}_{expr} \rangle + \alpha \text{GW}_{geo}(\mu_t, \mu_{t+1}, \pi) + \epsilon H(\pi) + \tau_1 \text{KL}(\pi \mathbf{1} \| p) + \tau_2 \text{KL}(\pi^T \mathbf{1} \| q)$$
   其中边缘先验 $p, q$ 根据细胞周期评分与凋亡指标基因表达进行动态缩放。

2. **流形空间测地线路径插值 (Geodesic Interpolation)**：
   传统流匹配使用欧氏直线插值路径，导致训练出的流场“穿透”物理屏障。本项目中，若最优传输矩阵将细胞 $i$ (起点为 $\mathbf{x}_0$) 与细胞 $j$ (终点为 $\mathbf{x}_1$) 耦合，我们利用 Delaunay 三角剖分图回溯出它们之间的空间测地线最短路径折线序列：$\Gamma_{ij} = \{\mathbf{s}_0, \mathbf{s}_1, \dots, \mathbf{s}_k\}$。
   构造常速测地线映射路径 $\gamma_{ij}(t): [0, 1] \to \mathcal{M}$，满足 $\gamma_{ij}(0) = \mathbf{x}_0$，$\gamma_{ij}(1) = \mathbf{x}_1$。

3. **联合隐空间状态与切矢量匹配目标**：
   我们将高维表达隐空间 $Z$ (无屏障约束) 与物理空间 $X$ (测地约束) 组合为联合状态向量 $\mathbf{s} = [\mathbf{z}, \mathbf{x}]$。
   在时间 $t \in [0, 1]$ 下，条件概率路径与目标漂移速度分别定义为：
   $$\mathbf{s}_t = \left[ (1-t)\mathbf{z}_0 + t\mathbf{z}_1, \; \gamma_{ij}(t) \right] + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$
   $$\mathbf{v}_{target}(t) = \left[ \mathbf{z}_1 - \mathbf{z}_0, \; \dot{\gamma}_{ij}(t) \right]$$
   其中 $\dot{\gamma}_{ij}(t) = \frac{d}{dt}\gamma_{ij}(t)$ 为空间测地线在时刻 $t$ 的切矢量。

4. **流匹配回归损失函数**：
   漂移速度场网络 $\mathbf{u}_\theta(\mathbf{s}, t)$ 通过最小化以下损失进行训练：
   $$\mathcal{L}_{GFM}(\theta) = \mathbb{E}_{t, \mathbf{s}_0, \mathbf{s}_1 \sim \pi^*, \epsilon} \left\| \mathbf{u}_\theta(\mathbf{s}_t, t) - \mathbf{v}_{target}(t) \right\|^2$$

*   **数理逻辑**：通过将插值路径限制在黎曼流形测地线上，神经网络在训练阶段即直接拟合弯曲的避障切向量。这使得神经网络训练完毕后，运行 ODE 积分时**无需任何人为添加的排斥势能场，轨迹即可自然绕行物理屏障**，极大地提升了模型的鲁棒性与理论完备性。

---

## 4. 技术路线与关键算法设计

整个 `SpaLineage-OT` 流程设计如下：

```mermaid
flowchart TD
    subgraph A[数据质控与速度估计]
        A1[时序空间转录组/单细胞数据] --> A2[形态学 H&E/IF 图像提取阻抗流形]
        A1 --> A3[scVelo 估计 RNA 速度向量 v]
    end

    subgraph B[数学物理几何建图]
        B1[胞质心 Delaunay 拓扑网]
        A2 --> B2[计算测地线阻抗距离 d_geo]
        A3 --> B3[构建非对称表达代价矩阵 C_expr]
    end

    subgraph C[非对称测地线最优传输求解]
        B2 & B3 & B1 --> C1[非对称 Fused Gromov-Wasserstein 求解]
        C1 --> C2[计算时间步对偶耦合矩阵 𝜋*]
    end

    subgraph D[连续流匹配与动力学模拟]
        C2 --> D1[薛定谔桥流匹配训练 Neural ODE]
        D1 --> D2[预测任意中间时间点细胞空间分布与轨迹]
    end
    
    D2 --> E[提取纤维化转变关键标志物与空间迁移速度 field]
```

### 4.1 数据预处理与阻抗映射
*   **测地线图构建**：基于 ST 切片的 DAPI 或 Collagen-I 免疫荧光强度，在网格坐标系上运行快速行进算法（Fast Marching Method, FMM）计算距离矩阵。
*   **RNA速度降维映射**：将高维 spliced/unspliced 表达矩阵通过 scVelo 降维，并在经过 Batch 校正后的共享潜在表征空间 $Z$ 中映射出相应的即时速度向量。

### 4.2 耦合矩阵的极化求解 (Polarized Sinkhorn)
由于 Fused Gromov-Wasserstein 属于非凸二次规划问题，在大规模单细胞级（$N > 10^5$）计算极慢。本项目采用基于**极化 Sinkhorn 算法 (Entropic Polarized Sinkhorn)** 求解，在每一步迭代中通过快速卷积加速核矩阵相乘，使得计算复杂度从 $O(N^3)$ 降至 $O(N \log N)$。

### 4.3 评估数据集与对比基线设计

为了定量评估 `SpaLineage-OT` (GI-USBFM) 算法的优越性，本课题设计了完整的基准测试（Benchmarking）方案：

#### 4.3.1 评估数据集
1. **高维合成几何数据集 (Synthetic Obstacles)**：
   在二维空间中布置具有复杂边界的物理屏障（如圆环形、马蹄形高阻抗壁），并构建分叉分化表达谱（Branching Lineage）。该数据集具有绝对的“轨迹真值（Ground Truth）”，用以验证算法避障及追踪分支拓扑的能力。
2. **时序小鼠肺损伤 Visium 数据集 (GSE198765)**：
   收集博来霉素（Bleomycin）诱导后 0, 3, 7, 10, 14, 21 天的密集的空间转录组切片。在纤维化进展过程中，成纤维细胞会向肌成纤维细胞分化，并被重度沉积 Jun（Collagen Plaques）所物理分隔，是检验“形态学阻抗流形测地线约束”的最佳现实病理数据集。
3. **小鼠胚胎发育时空图谱 (MOSTA)**：
   单细胞分辨率的 Stereo-seq 时序数据集（E9.5 - E12.5），涵盖数百万细胞，用于评估算法在超大规模数据集上的计算效率与命运谱系预测精度。

#### 4.3.2 对比基线 (Baselines)
为了凸显我们算法在“空间避障”、“流形动力学”和“时间非对称性”上的突破，对比以下四类主流算法：
*   **Moscot (spatiotemporal 模块)**：标准的 Fused Gromov-Wasserstein 算法，其空间度量仅采用欧氏距离，且不引入 RNA 速度引导。
*   **SpaTrack**：基于离散 OT 映射的空间谱系追踪算法，不具备连续时空常微分方程的流场外推能力。
*   **SOCS**：引入局部邻近 contiguous 约束的静态对齐算法，无法重建跨时间点的动态连续演化轨迹。
*   **CellRank 2 / WOT**：仅在转录组表达隐空间中推断状态转移概率，完全丢失物理空间维度。

#### 4.3.3 评估参数 (Evaluation Metrics)

我们将通过以下四个核心量化指标，证明本算法的优越性：

| 评估指标 | 数学定义 | 评估目的与物理意义 | 算法成功表现 |
| :--- | :--- | :--- | :--- |
| **表达重构瓦瑟斯坦距离 (EWD)** | $W_1(\hat{P}_{expr}(t), P_{expr}(t))$ | 在交叉验证（留一时间点法）中，评估模拟生成的细胞表达分布与真实观测分布的贴合度。 | **EWD 显着低于 Moscot 和 SpaTrack**，证明流匹配比欧氏插值更能拟合真实分化路径。 |
| **物理屏障越界率 (BCR)** | $\frac{N_{intersect}}{N_{total}} \times 100\%$ | 统计模拟轨迹中，直接穿透组织物理屏障（如致密胶原斑块）的轨迹占比。 | **BCR 贴近 0%**，而 Moscot 和 SpaTrack 会由于欧氏直线路径导致 BCR 偏高。 |
| **速度余弦一致性 (VCC)** | $\text{mean}_{i} \cos(\mathbf{u}_{spatial}(\mathbf{s}_i), \mathbf{v}_i^{proj})$ | 评估模型输出的即时空间速度矢量与细胞原位 RNA 速度投影方向的一致性。 | **VCC 显着提高**，表明模型成功将转录速率动能转化为空间位移。 |
| **细胞命运转移一致性 (CTC)** | $\sum_{(c_a, c_b) \in \text{valid}} \pi^*(c_a, c_b)$ | 统计最优传输耦合矩阵中，符合已知生物学转化规律（如成纤维 $\to$ 肌成纤维）的概率质量比例。 | **CTC 大幅增加**，非对称 VG-OT 减少了“逆分化”和“跨谱系乱配”的假阳性概率。 |

---


## 5. 课题可行性分析与资源预算

### 5.1 可行性分析
1. **算法优势 — 算力消耗极低（对比扩散模型）**：
   *   *为什么放弃课题二的 SCGD 扩散模型？* 扩散模型需要通过数千步逆向去噪，在像素级拟合连续坐标非常吃显存和 GPU（如前所述，至少需要 A100/RTX 4090）。
   *   *本课题算力占用*：本算法的底层核心是**最优传输矩阵求解与常微分方程（ODE）回归**。使用轻量级的多层感知机（MLP）或者图神经网络（GNN）即可参数化 Neural ODE，通过 TorchDiffEq 求解器进行极快的积分计算。在一块普通游戏显卡（如 RTX 3060 / 4060，**8GB 显存**）上，即可在 **1 小时内** 完成 10 万细胞级别的完整轨迹推断。
2. **公开数据集的充足性**：
   *   **小鼠肺纤维化时序数据集 (GSE198765)**：包含博来霉素（Bleomycin）诱导后 0, 3, 7, 10, 14, 21 天的 Visium 切片，时序极其密集，完美符合本课题要求。
   *   **小鼠肝损伤时序 Stereo-seq 数据集 (CNP0002890)**：提供碳氯化四（CCl4）肝损伤模型的超高分辨率时序切片，可用于方法学的跨平台验证。

### 5.2 软硬件配置与资源预算
*   **硬件要求**：
    *   CPU: Intel/AMD 8核及以上。
    *   内存 (RAM): 32GB（分块计算）或 64GB。
    *   GPU: NVIDIA 显卡（显存 $\ge$ 8GB，如 RTX 3060/4060/4070 均可），支持 CUDA 11.8+。
*   **软件依赖**：
    *   `PyTorch` $\ge 2.0$, `torch-geometric`
    *   `moscot` $\ge 0.2.0$, `ott-jax` (Optimal Transport Tools)
    *   `scvelo`, `scvi-tools`
    *   `torchdiffeq` (Neural ODE 求解包)

---

## 6. 参考文献

1.  **Klein, N., et al.** (2024). *Scalable single-cell optimal transport with moscot*. **Nature Methods** (or bioRxiv: 10.1101/2023.05.11.540284).
2.  **Schiebinger, G., et al.** (2019). *Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming*. **Cell**, 176(4), 928-943.
3.  **Bergen, V., et al.** (2020). *Generalizing RNA velocity to transient cell states through dynamical modeling*. **Nature Biotechnology**, 38(12), 1408-1414.
4.  **Zeira, R., et al.** (2022). *Alignment and integration of spatial transcriptomics data using optimal transport*. **Bioinformatics**, 38(7), 1920-1929.
5.  **Wang, J., et al.** (2024). *SpaTrack: reconstructing spatial-temporal cell lineages using optimal transport*. **Briefings in Bioinformatics**, 25(2), bbae043.
6.  **Chen, Y., et al.** (2025). *SOCS: Spatiotemporal Optimal transport with Contiguous Structures for spatial transcriptomics*. **bioRxiv** preprint.
7.  **Tong, A., et al.** (2023). *Improving single-cell optimal transport with Schrödinger Bridges*. **ICML 2023**.
8.  **Lipman, Y., et al.** (2023). *Flow Matching for Generative Modeling*. **ICLR 2023**.
9.  **Svensson, V., et al.** (2018). *Exponential scaling of single-cell RNA-seq in the literature*. **Nature Protocols**, 13(4), 599-604.
10. **Luecken, M. D., & Theis, F. J.** (2019). *Current best practices in single-cell RNA-seq analysis: a tutorial*. **Molecular Systems Biology**, 15(6), e8746.
11. **Huguet, G., et al.** (2023). *Manifold Interpolating Optimal-Transport Flows for Trajectory Inference*. **arXiv preprint arXiv:2306.02324**.
12. **De Bortoli, V., et al.** (2021). *Diffusion Schrödinger Bridge with Applications to Generative Modeling*. **NeurIPS 2021**.
13. **Chen, T., et al.** (2024). *GENOT: Geometry-Aware Entropic Optimal Transport for Generative Multi-Omic Mapping*. **ICML 2024**.
