# SpaLineage-OT 算法深度优化提案 (Algorithmic Optimization Proposals)

基于最优输运（Optimal Transport）与流匹配（Flow Matching）的前沿文献（如 *Riemannian Flow Matching* 与 *Geodesic Flow Matching*），我们为 **SpaLineage-OT** 设计了三个在保留核心创新点（Geodesic FGW + SBFM）的前提下，能显著提升定量重构精度和发育轨迹物理合理性的优化方案。

---

## 提案 1：测地线流匹配 (Geodesic Flow Matching, GFM) —— 解决轨迹“理论与实践脱节”

### 现状痛点 (The Gaps)
虽然我们在最优输运（FGW）匹配阶段使用了测地线距离（Geodesic Distance）来指导细胞配对，但在后期的**流匹配（Flow Matching）训练阶段**，代码仍采用 Euclidean 直线对配对细胞进行插值：
$$x_t = (1 - t)x_0 + t x_1$$
并以此直线的方向向量作为漂移网络（Drift MLP）的拟合目标：
$$v_{target} = x_1 - x_0$$
这导致训练出来的 Neural ODE 实际上是在模拟**直线运动**。尽管在空间边界处由于网络表达能力限制有一些弯曲，但它本质上没有“沿着流形弯曲通道运动”的先验约束。

### 优化方案 (The Solution)
将 Euclidean 插值替换为**流形测地线插值 (Geodesic Interpolation)**：
1. **路径提取**：在采样出细胞配对 $(x_0, x_1)$ 后，利用之前构建好的 Delaunay 测地线图，提取出两点之间的 Dijkstra 最短测地路径：
   $$\gamma = [p_0, p_1, \ldots, p_K] \quad (p_0 = x_0, p_K = x_1)$$
2. **时间参数化**：计算路径的总长度 $L_{total}$ 和各分段的累积长度。将时间 $t \in [0, 1]$ 映射到测地路径的累积长度上。
3. **漂移场目标**：
   * 在时间 $t$ 采样出测地线上的精确坐标 $x_t = \gamma(t)$。
   * 以该分段的瞬时方向向量（即测地线的切线切向量 $\dot{\gamma}(t)$）作为漂移网络的拟合目标：
     $$v_{target} = L_{total} \cdot \frac{p_{j+1} - p_j}{\|p_{j+1} - p_j\|}$$
4. **预期效果**：Neural ODE 将直接学习到如何在空间中“绕过障碍物”，其输出的连续轨迹将完美契合胚胎的几何轮廓，显著降低 Barrier Crossing Rate。

---

## 提案 2：自适应流形边界剪裁 (Adaptive Delaunay Edge Filtering) —— 解决固定阈值导致的拓扑断裂

### 现状痛点 (The Gaps)
目前算法通过一个固定的全局最大边长参数 `delaunay_max_edge_length = 50.0` 来过滤 Delaunay 三角形边。
* 如果胚胎中某些真实的过渡区域细胞分布较稀疏，其细胞间距大于 50，则这些区域会被**强行断开**（无法计算最短路径）。
* 如果设得太大（如 100），在密集的区域，原本是空腔的边界又会被**强行连通**（导致细胞穿过空腔）。

### 优化方案 (The Solution)
将全局固定阈值改为**局部自适应邻域阈值 (Adaptive Local Thresholding)**：
1. 对于 Delaunay 剖分生成的每条边 $(u, v)$，其允许的最大边长阈值 $T(u, v)$ 由节点 $u$ 和 $v$ 的局部 $k$-近邻平均距离 $d_{kNN}(u)$ 和 $d_{kNN}(v)$ 决定：
   $$T(u, v) = \sigma \cdot \max(d_{kNN}(u), d_{kNN}(v))$$
   其中 $\sigma$ 是一个尺度因子（如 2.5 或 3.0）。
2. **预期效果**：在细胞密集的区域，阈值自动变小，灵敏地识别出极小的组织空腔和血管通道；在细胞稀疏的边缘区域，阈值自动变大，确保胚胎整体流形的连通性，消除 Dijkstra 报错，并提高测地线距离矩阵的计算鲁棒性。

---

## 提案 3：物理先验正则化损失 (Physics-Prior Regularized Loss) —— 提升分化方向约束

### 现状痛点 (The Gaps)
当前的 Flow Matching 漂移网络训练是纯数据驱动的，只拟合起点和终点。虽然我们在 OT 匹配中融入了 RNA 速度（Velocity）引导，但漂移网络本身并不知道胚胎发育中实际观测到的瞬时分化方向（DPT 梯度或 RNA 速度）。

### 优化方案 (The Solution)
在 Flow Matching 的 Mean Squared Error 损失函数中，引入一个**物理正则化项 (Physics-Prior Regularization)**：
$$\mathcal{L} = \mathcal{L}_{FM} + \lambda \cdot \mathcal{L}_{physics}$$
$$\mathcal{L}_{physics} = \mathbb{E}_{x_t, t} \left[ 1.0 - \text{CosineSimilarity}\left( v_{\theta}(x_t, t), v_{phy}(x_t) \right) \right]$$
* 其中 $v_{\theta}$ 是网络预测的漂移速度，$v_{phy}(x_t)$ 是在空间坐标 $x_t$ 处的局部 RNA 速度向量或 DPT 伪时间梯度向量。
* $\lambda$ 是平衡系数（例如 0.1）。

### 预期效果
该正则化约束漂移场在任何空间位置不仅要指向终点细胞，其运动方向还要尽量贴合局部观测到的细胞发育分化物理梯度。这能极大地抑制反物理的异常漂移轨迹，使得预测的中间状态 E10.5 的转录组（PCA/表达谱空间）和空间位置在生物学上更具有可解释性。
