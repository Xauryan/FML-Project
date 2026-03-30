# 实验报告

## 一、任务概述

基于经典卷积神经网络 LeNet-5 实现 MNIST 手写数字识别，完成模型训练、性能评估、激活函数对比实验，并对模型架构、卷积核权重和特征图进行可视化分析。

## 二、模型架构

采用 LeNet-5 网络结构，各层参数如下：

| 层名 | 类型 | 输出尺寸 | 参数量 |
|------|------|----------|--------|
| conv1 | Conv2D(6, 5×5) | 24×24×6 | 156 |
| pool1 | AvgPooling(2×2) | 12×12×6 | 0 |
| conv2 | Conv2D(16, 5×5) | 8×8×16 | 2,416 |
| pool2 | AvgPooling(2×2) | 4×4×16 | 0 |
| flatten | Flatten | 256 | 0 |
| fc1 | Dense(120) | 120 | 30,840 |
| fc2 | Dense(84) | 84 | 10,164 |
| output | Dense(10, softmax) | 10 | 850 |

总参数量：44,426。

## 三、改进和优势

作业提供的模板代码（`references/assignment-starter/`）存在以下不足，本项目逐一进行了改进：

### 3.1 网络架构

原始模板使用的是自定义 CNN（Conv(32, 5×5) → MaxPool → Conv(64, 3×3) → MaxPool → Dense(128) → Dense(64)）。

本项目改为标准 LeNet-5（Conv(6, 5×5) → AvgPool → Conv(16, 5×5) → AvgPool → Dense(120) → Dense(84)），并在此基础上进行了以下优化：

- **MaxPooling 替代 AveragePooling**：MaxPooling 在边缘和笔画特征提取上表现更优，有助于区分形状相似的数字（如 3/5/8）
- **Dropout 正则化**：在 Flatten 后添加 Dropout(0.25)，在 fc1 后添加 Dropout(0.5)，有效减少过拟合
- **数据增强**：通过 Keras 预处理层实现 ±8° 随机旋转、±8% 随机平移和 ±8% 随机缩放，仅在训练时生效，显著提升模型对手写变体的鲁棒性

### 3.2 优化器

原始模板使用 `Adadelta()`，默认学习率下在 MNIST 上收敛较慢，10 个 epoch 仅能达到约 97% 的准确率。

本项目改用 `Adam(lr=0.001)`，同样 10 个 epoch 即可达到 99% 以上。同时引入 `ReduceLROnPlateau` 学习率调度器（验证损失停滞 2 个 epoch 后学习率减半），配合 `EarlyStopping`（patience=5）和更长的训练周期（20 个 epoch），使模型能充分收敛并在后期精细调优。

### 3.3 激活函数对比实验

原始代码仅使用单一激活函数，缺少实验对比。

本项目系统地训练了 tanh、relu、sigmoid 三组模型，独立记录训练曲线和测试指标，并绘制对比图表，直观展示不同激活函数对收敛速度和最终性能的影响。

### 3.4 可视化

原始模板没有可视化输出。

本项目实现了四类可视化：

- **模型架构图**：通过 `keras.utils.plot_model` 生成（graphviz 不可用时自动回退到 matplotlib 手绘方案）
- **卷积核权重热力图**：提取 conv1（6 个 5×5）和 conv2（16 个 5×5）的权重，使用 coolwarm 配色
- **特征图可视化**：对测试样本逐层提取 Conv 和 Pool 的输出，使用 viridis 配色
- **混淆矩阵**：基于 scikit-learn 计算，使用 seaborn 热力图渲染

### 3.5 数据集划分

原始代码直接将测试集同时用作验证集，可能导致模型选择偏向测试数据。

本项目从训练集中划出 10% 作为验证集（通过 `train_test_split` 分层抽样），使模型选择和超参调优更准确。

### 3.6 GUI

原始 GUI 依赖 `win32gui` 进行画布截图，仅能在 Windows 上运行。

本项目改用 PIL 离屏画布方案：在 Tkinter Canvas 绑定鼠标事件的同时，同步在 `PIL.ImageDraw` 上绘制，识别时直接从 PIL Image 中获取像素数据，无需平台相关的截图 API。

### 3.7 图像预处理

原始 GUI 直接将 300×300 画布缩放到 28×28，未对笔画进行居中处理。

本项目在 `src/preprocessing.py` 中实现了与 MNIST 数据集一致的预处理流程：裁切有效区域 → 等比缩放到 20×20 → 居中放置到 28×28 画布 → 反色归一化。这与 MNIST 原始数据的制作方式一致，显著提升了手写识别的准确率。

### 3.8 代码工程化

原始代码是单文件脚本，所有逻辑（数据加载、模型定义、训练、绘图）堆在一个文件中。

本项目将代码拆分为 `src/` 包，按职责划分模块（modeling / train / visualize / inference / preprocessing / fonts / paths），入口脚本为单行 wrapper，便于维护和测试。

### 3.9 Tauri 桌面应用

本项目在 Python GUI 之外，额外提供了基于 Tauri 的桌面应用。模型通过 tf2onnx 转换为 ONNX 格式，前端使用 onnxruntime-web 直接推理，无需用户安装 Python 环境。通过 GitHub Actions 自动构建 macOS / Windows / Linux 安装包并发布到 Releases。

## 四、激活函数对比结果

| 激活函数 | 测试准确率 | 测试损失 |
|----------|-----------|----------|
| tanh | 98.75% | 0.0391 |
| relu | **99.12%** | **0.0276** |
| sigmoid | 98.56% | 0.0433 |

ReLU 在收敛速度和最终准确率上均优于 tanh 和 sigmoid，与理论预期一致：ReLU 计算简单、不存在梯度饱和问题，在深层网络中表现更稳定。Sigmoid 表现最差，主要原因是梯度消失导致深层权重更新缓慢。

### 优化效果

相比优化前（无数据增强、AveragePooling、无 Dropout、10 个 epoch），优化后各激活函数的测试准确率均有提升：

| 激活函数 | 优化前 | 优化后 | 提升 |
|----------|--------|--------|------|
| tanh | 98.38% | 98.75% | +0.37% |
| relu | 98.81% | **99.12%** | +0.31% |
| sigmoid | 98.25% | 98.56% | +0.31% |

优化策略包括：数据增强（随机旋转、平移、缩放）、MaxPooling 替代 AveragePooling、轻量 Dropout 正则化（0.1/0.25）、ReduceLROnPlateau 学习率调度以及更长的训练周期（20 epoch, patience=5）。三种激活函数均获得了一致的提升。

以数字 3 为例，优化前后的混淆矩阵对比（ReLU 最佳模型）：

| | 优化前 | 优化后 |
|---|---|---|
| 正确识别 | 981/1010 | 1007/1010 |
| 准确率 | 97.1% | 99.7% |
| 误判总数 | 29 | 3 |

数据增强（随机旋转、平移、缩放）和 MaxPooling 的引入显著提升了形状相似数字（3/5/8）的区分能力。

## 五、可视化产出

训练和可视化脚本运行后，在 `outputs/` 目录下生成以下文件：

| 文件 | 内容 |
|------|------|
| `training_curves_*.png` | 各激活函数的训练/验证 loss 和 accuracy 曲线 |
| `activation_comparison.png` | 三种激活函数的验证指标对比 |
| `confusion_matrix.png` | 最佳模型在测试集上的混淆矩阵 |
| `model_architecture.png` | 模型架构图 |
| `filters_conv1.png` / `filters_conv2.png` | 卷积核权重热力图 |
| `feature_map_*.png` | 各卷积层和池化层的特征图 |
| `input_image.png` | 用于特征图分析的输入样本 |
| `model_summary.txt` | 模型参数摘要 |
| `activation_results.csv` | 激活函数对比数据 |
