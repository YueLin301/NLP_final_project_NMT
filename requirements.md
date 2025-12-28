这是一份来自**深圳河套学院 (Shenzhen Loop Area Institute)** 的 **2025年秋季 NLP 与大语言模型课程**  的期中与期末联合项目说明。

该项目的核心目标是**实现中文到英文的机器翻译 (NMT)**，并对比基于 **RNN** 和 **Transformer** 两种架构的性能与差异 。

以下是基于文档内容的详细作业拆解：

### 1. 截止日期与重要时间点

* 
**项目提交截止 (DDL):** 12月28日 。


* 
**展示时间 (Presentation):** 12月28日 。



---

### 2. 核心任务要求 (Assignment Requirements)

你需要完成三个主要部分的任务：

第一部分：基于 RNN 的神经机器翻译 (RNN-based NMT) 

你需要构建并训练一个 RNN 模型，具体要求如下：

* 
**模型架构:** 使用 **GRU** 或 **LSTM**。编码器 (Encoder) 和解码器 (Decoder) 都必须包含**两层单向 (two unidirectional layers)** 结构 。


* 
**注意力机制 (Attention):** 实现注意力机制，并研究不同对齐函数（**点积 dot-product, 乘性 multiplicative, 加性 additive**）对性能的影响 。


* 
**训练策略:** 比较 **Teacher Forcing** 和 **Free Running** 策略的效果 。


* 
**解码策略:** 比较 **贪婪解码 (Greedy)** 和 **束搜索 (Beam-search)** 的效果 。



第二部分：基于 Transformer 的神经机器翻译 (Transformer-based NMT) 

* 
**从头构建 (From scratch):** 搭建一个 Encoder-Decoder 结构的 Transformer 模型并在数据上从头训练 。


* 
**架构消融实验 (Architectural Ablation):** 比较不同的位置编码（如 **绝对位置 vs. 相对位置**）和归一化方法（如 **LayerNorm vs. RMSNorm**）。


* 
**超参数敏感性:** 测试不同 Batch Size、学习率 (Learning Rates) 和模型规模对翻译性能的影响 。


* 
**预训练模型微调:** 微调一个预训练语言模型（如 **T5**），并将其性能与你从头训练的模型进行对比 。



第三部分：分析与对比 (Analysis and Comparison) 

你需要综合对比 RNN 和 Transformer 模型，分析维度包括：

* 
**模型架构:** 串行 vs 并行计算，递归 vs 自注意力 。


* 
**训练效率:** 训练时间、收敛速度、硬件需求 。


* 
**翻译性能:** BLEU 分数、流畅度、准确度 。


* 
**扩展性与泛化:** 长句处理能力、低资源场景表现 。


* 
**实际权衡:** 模型大小、推理延迟、实现难度 。



---

### 3. 数据集与预处理 (Datasets & Preprocessing)

* 
**数据来源:** 通过 Piazza 平台获取 。


* 
**数据文件:** 包含4个 JSONL 文件：小训练集 (100k)、大训练集 (10k)、验证集 (500) 和测试集 (200) 。


* 
注意：如果算力有限，可以使用小训练集中的 10k 句对进行训练，但鼓励尝试使用大数据集 。




* 
**数据格式:** JSONL 文件每行包含一对平行语料 。


* **预处理要求:**
* 
**清洗:** 去除非法字符，过滤低频词，截断过长句子 。


* **分词 (Tokenization):**
* 
**英文:** 使用 NLTK 或统计子词分割方法（如 BPE, WordPiece）。


* 
**中文:** 使用 Jieba（轻量级）或 HanLP（高精度）。




* 
**词表构建:** 过滤低频词以控制词表大小 。


* 
**Embedding 初始化:** 推荐使用预训练词向量初始化并进行微调 。





---

### 4. 提交要求 (Submission Requirements)

你需要提交以下两样东西：

1. **源代码 (Source Code):**
* 提交到 GitHub 仓库 。


* 必须包含一个名为 `inference.py` 的**一键推理脚本**，方便助教测试 。


* 
提示：仓库链接需明确写在报告第一页 。




2. **项目报告 (Project Report):**
* 
**格式:** PDF，命名格式为 `ID_name.pdf` (例如 `250010001_Zhang San.pdf`) 。


* 
**提交位置:** Piazza 平台 。


* 
**内容包含:** 模型架构描述、代码实现解释、实验结果分析（需可视化）、个人总结 。


* 
注意：评分不仅仅基于最终的 BLEU 分数 。





---

### 5. 展示 (Presentation) - **非常重要！**

* 
**分组:** 全班分为18组，每组约7人。可以自由组队，未组队者将随机分配 。


* 
**形式:** 10分钟展示 + 5分钟问答 (Q&A) 。


* 
**要求:** 每组进行一次联合展示，但**每位学生必须单独提交自己的项目报告** 。



---

6. 评分标准 (Scoring Criteria) 

请注意，**展示 (Presentation)** 占据了总分的一半，是权重大头。

| 内容 (Content) | 分值 (Score) |
| --- | --- |
| **Presentation (小组展示)** | **50%** |
| Transformer NMT 实现与实验 | 25% |
| RNN NMT 实现与实验 | 15% |
| 比较分析与讨论 | 5% |
| 项目报告 (撰写质量) | 5% |

---

### 7. 评价指标 (Metrics)

* 主要使用 **BLEU-4** 分数进行评估 。



### 8. 参考资源

文档推荐了以下工具和论文作为参考：

* 
**教程:** PyTorch Seq2Seq 教程 。


* 
**工具:** Jieba (中文分词), SentencePiece (子词分词) 。


* **论文:**
* 
*Attention is All You Need* (Transformer) 。


* 
*Neural Machine Translation by Jointly Learning to Align and Translate* (Attention in RNN) 。


* 
*Exploring the Limits of Transfer Learning...* (T5) 。





**总结建议：**
这就意味着你需要兼顾代码实现的完整性（特别是 Transformer 的各种消融实验）和展示的质量。鉴于 Presentation 占 50%，请务必给 PPT 制作和演讲排练预留充足时间。