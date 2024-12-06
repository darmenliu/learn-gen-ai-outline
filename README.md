# 软件工程师-生成式AI学习路线图

## 第一部分：基础知识

### AI与生成式AI概述
- 人工智能基本概念
- AI及生成式AI
- 专家系统
- 搜索算法

### 机器学习基础及理论
- 机器学习基础理论
- 模式识别和模式分类
    - 模式分类的基本原理
        - 决策理论
        - 分类器的作用
        - 模式识别的基本过程
    - 特征提取和选择
        - 特征提取
            - 特征空间的构建
            - 特征的表示与编码
            - 特征提取技术
                - 主成分分析
        - 特征选择
            - 特征重要性的评估方法
            - 特征选择算法
    - 分类方法
        - 统计分类
            - 贝叶斯分类器
            - Fisher线性判别分析（LDA）
            - 支持向量机（SVM）
        - 距离度量与非参数方法
            - 最近邻分类器（k-NN）
            - 距离度量（欧氏距离、曼哈顿距离、余弦相似度等）
    - 聚类与无监督学习
        - 聚类
            - k-Means聚类
            - 层次聚类
            - DBSCAN聚类
        - 无监督学习
            - 主成分分析（PCA
            - 自编码器（AutoEncoder）
            - 独立成分分析（ICA）
            - 隐变量模型
            - 非负矩阵分解（NMF）
    - 评估与优化
        - 评估指标
            - 准确率、召回率、F1值
            - ROC曲线、AUC值
        - 交叉验证
        - 超参数调优
            - 网格搜索
            - 随机搜索
            - 贝叶斯优化
            - 模型正则化技术（L1/L2正则化）
    - 贝叶斯方法与概率模型
        - 贝叶斯方法
            - 贝叶斯定理
            - 贝叶斯网络
        - 概率模型
            - 隐马尔可夫模型（HMM）
            - 高斯混合模型（GMM）
    - 异常检测与异常分类
    - 多模态模式识别
    - 图像识别
- 机器学习基础算法
    - 决策树
    - 贝叶斯网络
    - 回归分析
    - 支持向量机
    - K近邻算法
    - 遗传算法
    - 集成学习
        - Bagging方法（如随机森林）
        - Boosting方法（如AdaBoost、XGBoost）

### 神经网络基础
- 神经网络基本原理
    - 模拟生物神经系统的计算模型
    - 神经元模型
        - 神经元
        - 权重
        - 偏置
        - 激活函数
        - 损失函数
    - 神经网络的层次结构
        - 输入层
        - 隐藏层
        - 输出层
    - 常见激活函数
        - Sigmoid
        - Tanh
        - ReLU（Rectified Linear Unit）
        - Leaky ReLU
        - Softmax
    - 损失函数
        - 均方误差（MSE）
        - 交叉熵损失函数
- 神经网络训练
    - 前向传播
    - 反向传播
    - 优化算法
        - 梯度下降
        - 随机梯度下降
        - Adam
        - RMSprop
        - Adagrad
        - Momentum
- 正则化
    - L1正则化
    - L2正则化
    - Dropout
    - Batch Normalization

- 深度学习
    - 前馈神经网络（Feedforward Neural Network, FNN）
    - 卷积神经网络（Convolutional Neural Network, CNN）
    - 循环神经网络（Recurrent Neural Network, RNN）
    - 自编码器（Autoencoder）
    - 生成对抗网络（Generative Adversarial Network, GAN）
    - 图神经网络（Graph Neural Network, GNN）
    - 注意力模型（Attention Mechanism
    - 深度强化学习模型


## 第二部分：大语言模型(LLM)

### 什么是大语言模型
- 基于深度学习的自然语言处理模型
- 能够生成、理解和操作自然语言
- 典型模型：GPT（OpenAI）、BERT（Google）、LLaMA（Meta）

### Transformer架构
- 基础构件：多头注意力机制、前馈网络、位置编码
- 关键优点：并行处理长序列数据
- 变体：BERT、GPT、T5等

### 预训练与微调
- 预训练：在大规模语料库上训练语言模型
- 微调：针对特定任务的优化
- 自监督学习：掩码语言模型（MLM）、因果语言模型（CLM）

### 注意力机制
- 点积注意力（Scaled Dot-Product Attention）
- 自注意力与全局依赖建模
- 多头注意力的效果与优化

### 大语言模型的训练
#### 训练数据
- 数据来源：文本语料库、网页爬取内容、对话记录
- 数据清洗与标注
- 数据增强技术

#### 训练策略
- 监督学习：使用标注数据训练
- 半监督学习：结合标注和非标注数据
- 强化学习：如使用奖励模型优化（RLHF, 人类反馈强化学习）

#### 模型训练资源
- 大规模分布式计算
- GPU/TPU硬件加速
- 并行与分布式训练技术（如数据并行、模型并行）

#### 大模型的部署和维护
    - 本地部署
    - 云端分布式部署
    - API 及服务

## 第三部分：生成式AI应用

### Prompt Engineering
- 什么是提示词工程
- 提示词的基本结构
- 提示词工程技术分类
    - zero shot
    - few shot
    - chain of thought
    - React prompt
    - tree of thought
    - prompt chain
    - Reflexion
    - Self-Consistency

### RAG (Retrieval-Augmented Generation)
- RAG模型架构
    - 检索
    - 生成
- 向量数据库
- 知识库
- RAG应用
    - 问答系统
    - 智能客服
    - 文本生成
    - 代码生成
    - 多模态生成
- RAG 编程框架
    - langchain
    - llamaindex
### AI Agent
- Agent架构模式
    - Agent 基本架构
    - 工具
    - 任务规划
    - 多Agent协作
    - 记忆与知识库
    - Agent评估方法
- 多Agent体
- Agent 编程框架
    - langchain
    - autogen
    - openAI

### 文本生成应用
- 对话系统
- 文本摘要
- 代码生成
    - Code Pilot
    - Code Agent
- 创意写作
- 多语言翻译

### 多模态生成
- 文本到图像生成
- 图像到文本描述
- 视频生成
- 音频生成
- 跨模态转换
- OCR与语音识别

## 第四部分：工程实践

### 常用的提示词模板


### 平台及工具

#### 智能体低代码

- COZE
- DIFY

#### 写代码

- Github Copilot
- Cursor
- Codeium
    - Windsurf

#### 图像及视频生成
- Madejourney
- Stable Diffusion
- Civitai
- Flux

#### 音乐生成

- suno
- udio
- 

### 大语言模型

- OpenAI models
    - GPT4
    - GPT4o
- Claude models
    - claude 3.5
- Google models
    - Gemini 1.5
- Aliyun
    - qianwen 2.5
- Deepseek
    - deepseek 2.5
- Llama models
    - llama 3.1
    - llama 3.2
- Mixtral
    - mixtral 

### 图像生成模型

- DALL-E
- Stable diffusion models
- Flux models

### 音乐生成模型

[AI 工具集](https://ai-bot.cn/)
[参考链接: Generative AI Handbook](https://genai-handbook.github.io)

