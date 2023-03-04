---
title: "图神经网络的表达能力"
mathjax: true
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

同样，对于图神经网络算法，也存在这种情况。例如某些特定的图结构天然就无法被GCN等图神经网络算法区分，再比如，over-smoothing 导致多层图神经网络对于同一连通分支内的节点丧失区分度，这也是GNNs表达能力的问题。因此当你在使用GNN算法时发现效果不尽理想，不如仔细思考是否是你使用的GNN模型本身的表达能力有限导致你的数据无法很好地被处理。

近些年来也有若干研究GNN算法VC维或者泛化误差的文章，我将在文末列出，它们不在本博客的讨论范围。

### I. Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning [AAAI 2018]

:sunny: **重要定理**（简化版本）

**Lemma 1**:  Define Graph Laplacian of GCN as $\tilde{L}_{sym}=D^{-\frac{1}{2}}LD^{-\frac{1}{2}},L=D-A$, graph convolution in GCN (i.e. $H=D^{-\frac{1}{2}}AD^{-\frac{1}{2}}X$) equals to the Laplacian smoothing as $H=(I-\alpha \tilde{L})X$ (where alpha controls the weight of self attributes and neighbors') 

if let  $\alpha=1,\tilde{L}=\tilde{L}_{sym}$; i.e. **the graph convolution in GCN is a special form of Laplacian smoothing – symmetric Laplacian smoothing.** Note that $A$ above is the adjacency matrix of the graph including self-looping. 

> 上述lemma是显而易见的，因此在原文中并不是展示为lemma，而是文字描述，这里为了更加清晰，将其描述为lemma。

**Theorem 1**

![]({{ site.url }}{{ site.baseurl }}/assets/images/gnnpower-1.png)

> Lemma 1指明了GCN和拉普拉斯平滑之间的关系，而Theorem 1指明了对同一个矩阵做多次拉普拉斯变换会造成 over-smoothing的问题，原本节点本身的信息会被平滑掉，变换后矩阵保留的信息很有限。但要注意的是上述定理描述的场景和多层GCN还是有区别的，上述**定理中没有引入非线性变换项**，因此按照原文，我们只能说：“ **it raises potential concerns about stacking many convolutional layers in a GCN** ”。

### II: How powerful are graph neural networks [ICLR 2019]

 :bulb: 论文总结：

- sum aggregator is better than the mean/max one
- 在区分不同的图结构方面，Aggregation-based GNN 最多和 WL-test一样优
- 提出了GIN，在可数空间下，在区分不同图结构方面可以做到和WL-test一样优

本文的定理其实都比较直观，因此笔者可以用相对简洁的语言将其描述清楚，如下：

首先是全文对于GNN 结构的假设：![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-04-21-22-25-image.png)

:sunny: 重要定理：

**Lemma 1：** 假设G1，G2是两个非同构图（not isomorphic）. 如果一个图神经网络模型 $\mathcal{A}:\mathcal{G}\rightarrow \mathbb{R}^d$ 将G1 G2映射为不同的向量, the Weisfeiler-Lehman graph isomorphism test（WL-test） 也可以判断出两个图为非同构的。

> Lemma 1 表明了**在区分不同图结构方面**，Aggregated-based GNN至多和WL-test一样优；其实直觉上，就会发现WL-test和spatial GNN算法的思想是及其相似的，两者都不断地利用邻居节点的信息更新中心节点的信息，因此有如上定理也并不奇怪。
> 
> :exclamation: **需要注意的是 WL-test 本身也并不能做到区分所有的图结构；**

**Theorem 1:**   当 2.1 式中的COMBINE函数和AGGREGATE函数都是 **单射（injective)** 的，且graph-level readout function 也是  **单射（injective)** 的时，该GNN模型可以将任何WL-test判断非同构的两个图 G1 G2 映射为不同的向量。

> Theorem 1 给定了在何种条件下，GNN可以和WL-test达到一样的power；
> 
> 另外为了更加简洁，这里没有额外引入原文中 multiset 的概念，但是这个概念是比较重要的；简单来说，就是 **AGGERAGE函数的输入（即所有邻居节点的embedding）是不区分邻居节点顺序的**，是一个可能有重复值的集合（即multiset）。

:star2: 接下来的分析基于如下​重要假设：

- input features space X 是可数的（意味着输入特征不能是连续空间内的向量，限制很大）
- 邻居节点个数有限

基于Theorem 1，作者提出了 GIN，GIN保证了在上述假设中是和WL-test一样优的；GIN的模型结构如下，其中READOUT函数取求和函数 sum：

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-04-21-59-36-image.png)

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-04-21-59-47-image.png)

> 乍一看，4.1式和Theorem 1是有差距的，最重要的差距在于AGGEREGATE函数仍然是求和，而求和函数即使在可数空间下，也不会是单射的。但是GIN仍然满足Theorem1的原因在于，最外层的多层MLP是可以模拟复合函数的，换言之，4.1式原本的样子应该是 $\phi((1+\epsilon)f(h_v)+\sum f(h_u))$ ，MLP模拟了复合函数$f\circ g$ 。
> 
> 4.2 式借鉴了 Jumping Knowledge Networks，是为了增强模型效果，和单射无关

其实设计的关键就在于 单射函数 的设计，而且这个函数最好是足够简洁和易学习的，在输入的特征是连续空间的情况下，设计会更加复杂，因此作者考虑了在可数空间的情况；

最后，我们可以回顾一下GCN和GIN的区别，其实主要在两个方面：

- GCN采用一层前馈网络而不是MLP

- readout function不同， GIN readout function为 求和函数，而不是平均或者取最大值。

因此说了这么多，GIN对GCN的改进就体现在这两点上，改进其实都不大，而且GIN还只能保证对于可数空间下的优势，在连续空间的效果还未知；

相比较而言，这篇文章最大的价值其实体现在lemma 1和Theorem 1上，后续有许多研究致力于使GNN突破WL-test这一个上界，并且取得了许多不错的进展。

### Break the Ceiling: Stronger Multi-scale Deep Graph Convolutional Networks [NeurIPS 2019]

太难读, 遂放弃

### III. Graph Neural Networks Exponentially Lose Expressive Power For Node Classification [ICLR 2020]

 :bulb: 论文总结：

- information loss of GCN：
  
  每增加一层 GCN，hidden embedding以 $\lambda s$ 的速度靠近（当$\lambda s>1$时实际上是远离） invariant subspace $\mathcal{M}$ , 在 $\mathcal{M}$ 中，相同连通区域内且度相同的点具有相同的embedding，无法被区分开来
  其中 $\lambda$ 与邻接矩阵的性质有关，S是 GCN的所有权重矩阵 W 奇异值中的最大值；**因此，当  $\lambda s<1$ 时，随着网络层数增加，GCN 的 information loss 会不断严重** 

- 当graph 不是特别稀疏（not extremely sparse）且 节点个数足够多时， $\lambda$ 非常小并且随着网络层数增加趋向于0，此时S只有趋向于无穷时才能保障没有information loss；因此在这种情况下，'‘most GCNs suffer from information loss’‘

:star2: 重要假设：

- 模型结构：GCN with Relu 
- 没有考虑 residual link

这篇文章有一个比较大的令人担忧的点，文章无法解释sparse graph的over-smoothing情况，而且根据本文的推论，sparse graph是有利于减缓over-smoothing的；同时，作者在实验部分又提到Cora数据集要满足information loss是不够稠密的，因此需要手动增加边。这和实证研究中发现的问题有所矛盾：多篇paper已经表明，对于Cora数据集，over-smoothing问题同样明显。

因此本文提出的定理可以很好地解释dense graph的over-smoothing问题，并且给出了information loss的速度，但是对于sparse graph无法很好地解释，令人担心相关定理的证明是否不够 tight。