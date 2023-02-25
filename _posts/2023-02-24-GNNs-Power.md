---
title: "图神经网络的表达能力"
categories:
  - Blog
tags:
  - paper reading
  - GNN
  - machine learning theory
toc: true
---
# GNNs Power

本篇博客旨在总结近些年来对于图神经网络(GNN)表达能力的相关研究，为实际运用GNN算法时出现的各类问题指明原因和可能的解决办法。需要注意的是表达能力（expressive power）和机器学习理论中常讨论的泛化能力 (generalization) 完全不是一回事。你可以理解为前者在讨论算法本身在训练集上的拟合能力，后者讨论的是模型在测试集上的泛化误差等。例如，感知机（i.e. 一层单神经元的神经网络）可以实现与、或、非计算，而不能实现异或计算，这是模型本身表达能力的问题，而不是泛化性能的考虑范围。

同样，对于图神经网络算法，也存在这种情况。例如某些特定的图结构天然就无法被GCN等图神经网络算法区分，再比如，over-smoothing 导致多层图神经网络对于同一连通分支内的节点丧失区分度，这也是GNNs表达能力的问题。因此当你在使用GNN算法时发现效果不尽理想，不如仔细思考是否是GNN模型本身的表达能力不够导致你的数据无法很好地被处理，而不是盲目地认定是图数据的问题。

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

  每增加一层 GCN，hidden embedding以 $\lambda s$ 的速度靠近（或远离） invariant subspace $\mathcal{M}$ , 在 $\mathcal{M}$ 中，相同连通区域内且度相同的点具有相同的embedding，无法被区分开来

  其中 $\lambda$ 与邻接矩阵的性质有关，S是 GCN的所有权重矩阵 W 奇异值中的最大值；**因此，当  $\lambda s<1$ 时，随着网络层数增加，GCN 的 information loss 会不断严重** 

- 当graph 不是特别稀疏（not extremely sparse）且 节点个数足够多时， $\lambda$ 非常小并且随着网络层数增加趋向于0，此时S只有趋向于无穷时才能保障没有information loss；因此在这种情况下，'‘most GCNs suffer from information loss’‘

:star2: 重要假设：

- 模型结构：GCN with Relu 
- sparse graph 可能不受information loss的影响
- 没有考虑 residual link