# NeurIPS 2024 | 突破图数据挖掘边界，TEG-DB 开启文本图新篇章！

本文旨在简要介绍近期发表在 NeurIPS 2024 上的首个 Textual-Edge Graphs 的 Benchmark：TEG-DB (A Comprehensive Dataset and Benchmark
of Textual-Edge Graphs)。
![](http://skhzmchvj.hn-bkt.clouddn.com/tegdb_paper.png?e=1727513354&token=UmxhkwnGSn42CQhrWY1V06wEvHueyg7zW6CJsNKx:Gtyy2oIGD84aDv1bTIKVEBDoJeM=)

论文题目：

TEG-DB: A Comprehensive Dataset and Benchmark of Textual-Edge Graphs

论文地址：

https://arxiv.org/abs/2406.10310

代码地址：

https://github.com/Zhuofeng-Li/TEG-Benchmark

## Introduction

文本属性图（Text-Attributed Graphs）是一种在节点上有丰富文本信息的图结构，TAGs 广泛应用于社交网络 (social network)、引用网络 (citation network) 和推荐系统 (recommendation system) 等实际场景中。由于其强大且通用的表达能力，该领域近年来得到了快速发展。

![](http://skhzmchvj.hn-bkt.clouddn.com/tegdb_example.png?e=1727572382&token=UmxhkwnGSn42CQhrWY1V06wEvHueyg7zW6CJsNKx:zLpjcRdUMz67-g7NBPOURdzpJWo=)

然而目前 TAGs 面临三大挑战：1. 大多数 TAGs 数据集仅在节点上包含文本信息，而边的信息往往被简化为二元或分类属性。边文本 (edge text) 的缺乏限制了我们对实体间复杂关系的深入理解，阻碍了图数据挖掘技术的进一步发展；2. 数据格式，实验设置不统一，难以进行模型之间的比较；3. 由于缺乏全面的基准测试和分析，我们对图模型处理边文本信息能力仍然未知。

**为了解决这一问题，我们推出了TEG-DB —— 一个全面的以文本边为中心的图数据集和基准测试 (A Comprehensive Dataset and Benchmark
of Textual-Edge Graphs)。** 其主要有三个特点：1. TEG-DB datasets 提供了涵盖 4 个领域 9 个统一格式的 TEG 数据集，规模从小到大不等，均包含丰富的节点和边的原始文本数据。这些数据集填补了 TEGs 领域的空白，旨在为相关研究提供重要数据集资源。2. 我们开发了 TEGs 研究的标准化流程，涵盖数据预处理、加载和模型评估等关键阶段。3. 我们进行了广泛的基准实验，并对基于 TEGs 的方法进行了全面分析，深入探讨了不同模型及不同规模 pre-trained language models (PLMs) 生成的嵌入的效果、在 GNNs 中使用分离和交织嵌入方法 (seperate and entangled embedding methods) 的影响、边文本的作用以及不同领域数据集的影响。

## TEG-DB Datasets

为了构建同时满足节点和边具有丰富文本信息的数据集，我们选择了来自不同领域和规模的 9 个数据集。具体包括 4 个来自 Goodreads 的图书推荐领域用户-书籍评论网络，2 个来自 Amazon 的电商购物网络，1 个来自 Semantic Scholar 的学术引文网络，以及 2 个来自 Reddit 和 Twitter 的社交网络。数据集统计见下表。

![](http://skhzmchvj.hn-bkt.clouddn.com/tegdb_datasets.png?e=1727572408&token=UmxhkwnGSn42CQhrWY1V06wEvHueyg7zW6CJsNKx:1bqc0WSqGAq3axBmT7XP4vNiN60=)

## TEG Methods

**基于 PLM 的范式。** PLM 通过大规模文本训练，能够理解词语、短语和句子的语义关系和上下文。基于 PLM 的方法首先将 TEG 中节点和边的文本通过 PLM 进行嵌入表示 (embed)，例如对于节点 $u$，通过 embed 其自身以及所连接的边文本，我们可以得到 embedding $\boldsymbol{h}_u^{(0)}$ 作为节点 $u$ 初始化特征 (feature) 。之后我们使用多层感知器（MLP）整合 TEG 中的语义信息，获得最终的节点表征。公式如下：

$$
\begin{aligned}
\boldsymbol{h}_u^{(k+1)} &= \mathrm{MLP}_{\boldsymbol{\psi}}^{(k)}\left(\boldsymbol{h}_u^{(k)}\right) \\
\boldsymbol{h}_u^{(0)} &= \operatorname{PLM}(T_{u}) + \sum_{v \in \mathcal{N}(u)} \operatorname{PLM}(T_{e_{v, u}})
\end{aligned}
$$

其中，$\boldsymbol{h}_u^{(k)}$ 表示第 $k$ 层 MLP 中节点 $u$ 的表示，$T_u$ 和 $T_{e_{v,u}}$ 分别为节点 $u$ 和边 $e_{v,u}$ 的原始文本，节点 $v$ 是 $u$ 的邻居，$\psi$ 为MLP的可训练参数。

**尽管PLM显著提升了节点的表征能力，但由于未考虑 TEG 拓扑结构，限制了其对 TEG 中完整语义信息的捕捉。**

**基于 Edge-aware GNN 的范式。** GNN 通过消息传递 （message passing）来提取图结构中有意义的表征信息，具体定义如下：

$$
\begin{aligned}
\boldsymbol{h}_u^{(k+1)}&=\operatorname{UPDATE}_{\boldsymbol{\omega}}^{(k)}\left(\boldsymbol{h}_u^{(k)}, \operatorname{AGGREGATE}_{\boldsymbol{\omega}}^{(k)}\left(\left\{\boldsymbol{h}_v^{(k)}, \boldsymbol{e}_{v, u}, v \in \mathcal{N}(u)\right\}\right)\right)
\end{aligned}
$$

其中，$\boldsymbol{h}_u^{(k)}$ 表示GNN第 $k$ 层中节点 $u$ 的表示，初始特征向量 $\boldsymbol{h}_u^{(0)}$ 通过使用 PLM 对节点的原始文本进行 embed 获得。$e_{v, u}$ 表示从节点$v$ 到节点 $u$ 的边，其特征 $\boldsymbol{e}_{v, u} $ 同样由PLM对于边的原始文本进行 embed 得到。$k$ 代表GNN的层数，$\mathcal{N}$ 表示邻居节点集合，$u$ 为目标节点，$\boldsymbol{\omega}$ 为GNN中的学习参数。

然而，这种方法存在两个主要问题：**（1）现有的图机器学习方法如 GNN 对于边通常基于连通性（即二元属性表示是否有连接）和边属性（如类别或数值属性）进行操作，而非基于文本属性。** 然而在 TEG 中，边包含了丰富的文本，这便导致 GNN 远不足以处理这些复杂的文本信息所产生的语义关系。**（2）基于 GNN 的方法在捕捉节点以及边文本的上下文语义方面存在局限性。** 在 TEG 中，边和节点的文本通常交织在一起，在嵌入过程中将它们分别进行嵌入表示 (seperate embedding)，可能导致相互依赖关系信息的丢失，从而削弱 GNN 在整个消息传递过程中的有效性。

**基于 Entangled GNN 的范式。** 传统 GNN 方法将边和节点文本分离进行嵌入 (seperate embedding)，可能导致大量信息损失，特别是在 TEG 中。例如，在一个 citation network 中，每个节点表示一篇论文，一条边可能表示某篇论文引用、批评或使用了另一篇论文的某一部分。因此，边文本是不能独立于论文节点存在的，这便对节点以及边 seperate embedding 方法提出挑战。为避免文本嵌入后节点和边交互时的信息丢失，我们提出了一种新的方法，先将边文本和节点文本 Entangle 在一起，再进行 embed，作为节点的初始化 embedding。随后对节点进行消息传递操作。该方法的公式如下：

$$
  \begin{aligned}
\boldsymbol{h}_u^{(k+1)}&=\operatorname{UPDATE}_{\boldsymbol{\omega}}^{(k)}\left(\boldsymbol{h}_u^{(k)}, \operatorname{AGGREGATE}_{\boldsymbol{\omega}}^{(k)}\left(\left\{\boldsymbol{h}_v^{(k)}, v \in \mathcal{N}(u)\right\}\right)\right) \\
{h}_u^{0}&= \operatorname{PLM}(T_u, \{T_v, T_{{e}_{v, u}}, v \in \mathcal{N}(u)\})
  \end{aligned}
$$

其中，$\boldsymbol{h}_u^{(k)}$ 表示 GNN 第 $k$ 层中节点 $u$ 的表示。$T_v$、$T_u$ 和 $ T*{e*{v,u}}$ 分别表示节点 $v$、节点 $u$ 及其连接边的原始文本。 $k$ 为GNN的层数，$\mathcal{N}$表示邻居节点集合，$u$为目标节点，$\boldsymbol{\omega}$为GNN中的学习参数。

**相比于现有方法，该方法的优势在于能够有效保留节点与边之间的语义关系，更适合捕捉复杂的关系。**

**LLM as Predictor 的范式。** 利用 LLM 强大的文本理解能力，LLM 可以直接被用于解决图级别问题。具体而言，我们为每个数据集采用一个包含相应的节点和边文本的 text prompt，从而让 LLM 回答特定问题，例如节点分类或链接预测。我们可以正式定义如下：

$$
\begin{aligned}
    A = f\{\mathcal{G}, Q\}
\end{aligned}
$$

其中，$f$ 是提供图信息的 prompt，$\mathcal{G}$ 表示一个TEG，$Q$ 为问题。

## TEG Experimental Results

### Baselines

- 在基于PLM的范式中，我们使用三种不同规模的 PLM 对节点文本进行编码，以生成节点的初始嵌入。这三种模型分别是：大模型 GPT-3.5-TURBO，中型模型 Bert-Large，以及小型模型 Bert-Base。
- 在基于 Edge-aware GNN 的范式中。我们选择了五种流行的 GNN 模型：GraphSAGE、GeneralConv、GINE、EdgeConv 和 GraphTransformer。我们使用与 PLM 范式相同的三种规模的PLM对节点和边的文本进行编码，之后这些文本嵌入作为节点和边的初始特征。
- 在 LLM as Predictor 的范式中。我们选择通过 API 访问 GPT-3.5-TURBO 和 GPT-4，以平衡性能和成本。

### Node Classification

**下表展示了不同数据集上节点分类在中的效果：**
![](http://skhzmchvj.hn-bkt.clouddn.com/tegdb_node_classification_results.png?e=1727512448&token=UmxhkwnGSn42CQhrWY1V06wEvHueyg7zW6CJsNKx:89cVuPpDwnHPfIuxQvXTJXQFIVQ=)

### Link Prediction

**下表展示了不同数据集上链接预测的效果：**
![](http://skhzmchvj.hn-bkt.clouddn.com/tegdb_link_prediction_results.png?e=1727512426&token=UmxhkwnGSn42CQhrWY1V06wEvHueyg7zW6CJsNKx:9Cj9Ts0YSQxPMokzVXzp5e3iiT8=)
