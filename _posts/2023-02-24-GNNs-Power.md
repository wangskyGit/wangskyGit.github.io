---
title: "图神经网络的表达能力"
categories:
  - papers
tags:
  - GNN
  - machine learning theory
---
# GNNs Power

### I: How powerful are graph neural networks [ICLR 2019]

 :bulb: 论文总结：

- **sum aggregator is better than mean/max**
- 在区分不同的图结构方面，Aggregation-based GNN 最多和 WL-test一样优

:star2: ​重要假设：

- input features space X 是可数的（意味着输入特征不能是连续空间内的向量，限制很大）
- 邻居节点个数有限

### II. Graph Neural Networks Exponentially Lose Expressive Power For Node Classification [ICLR 2020]

 :bulb: 论文总结：

- information loss of GCN：

  每增加一层 GCN，hidden embedding以 $\lambda s$ 的速度靠近 invariant subspace $\mathcal{M}$ , 在 $\mathcal{M}$ 中，相同连通区域内且度相同的点具有相同的embedding，无法被区分开来

  其中 $\lambda$ 与邻接矩阵的性质有关，S是 GCN的所有权重矩阵 W 奇异值中的最大值；**当  $\lambda s<1$ 时，随着层数增加，GCN 的 information loss 会更加严重** 

- 当graph 不是特别稀疏（not extremely sparse）且 节点个数足够多时， $\lambda$ 非常小，S趋向于无穷时才能保障没有information loss；因此在这种情况下，'‘most GCNs suffer from information loss’‘

:star2: 重要假设：

- 模型结构：GCN with Relu 
- sparse graph 可能不受information loss的影响
- 没有考虑 residual link