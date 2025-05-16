# 光谱-图像恒星分类 (Spectrum-Image Star Classification)

本项目旨在利用深度多模态网络，通过结合恒星的光谱数据和测光图像数据，对 M 型恒星进行分类。

## 项目结构

```
.
├── checkpoints/         # 保存训练好的模型检查点 (.pth 文件)
├── images/              # 存储恒星的测光图像 (按子类分类, 如 M0/, M1/, ...)
├── plots/               # 保存训练过程中的指标图 (.png 文件)
├── spectra/             # 存储恒星的光谱数据 (.fits 文件, 按子类分类)
├── objects.csv          # 包含恒星目标信息和标签的 CSV 文件
├── model.py             # 定义多模态融合模型 (FusionModel)
├── loss_sic.py          # 实现光谱-图像对比损失 (SIC Loss)
├── train.py             # 训练脚本，包含数据加载、训练、验证、评估和历史记录
├── training_history.json # 记录每次训练的配置、逐轮指标和最终结果
└── README.md            # 本说明文件
```

## 依赖项

主要的 Python 依赖项如下：

*   `torch`: PyTorch 深度学习框架
*   `torchvision`: PyTorch 的计算机视觉库
*   `pandas`: 用于数据处理，特别是读取 CSV 文件
*   `numpy`: 用于数值计算
*   `Pillow`: 用于图像处理
*   `tqdm`: 用于显示进度条
*   `matplotlib`: 用于绘制训练指标图
*   `astropy`: 用于读取 FITS 格式的光谱文件
*   `scikit-learn`: 用于数据分割、标签编码和评估指标计算

建议使用 `pip` 或 `conda` 创建虚拟环境并安装这些依赖项。例如，使用 pip：

```bash
pip install torch torchvision pandas numpy Pillow tqdm matplotlib astropy scikit-learn
```

*注意：请根据你的系统和 CUDA 版本安装合适的 PyTorch 版本。*

## 数据准备

1.  **CSV 文件**: 确保根目录下有 `objects.csv` 文件，包含 `name`, `subclass`, `plate`, `mjd`, `fiberID` 列。`subclass` 应包含 M 型恒星的子类（例如 'M0', 'M1', ..., 'M4'）。
2.  **光谱数据**: 将 FITS 光谱文件放入 `spectra/` 目录下，并根据 `objects.csv` 中的 `subclass` 分类存放。例如，M0 型恒星的光谱应放在 `spectra/M0/` 目录下。文件名格式应为 `spec-pppp-mmmmm-ffff.fits`（p=plate, m=mjd, f=fiberID）。
3.  **图像数据**: 将 JPG 图像文件放入 `images/` 目录下，同样根据 `subclass` 分类存放。文件名格式应为 `{name}_image.jpg`，其中 `{name}` 对应 `objects.csv` 中的 `name` 列。

*脚本在启动时会检查 `objects.csv` 中列出的每个样本对应的光谱和图像文件是否存在，只使用文件齐全的样本进行训练。*

## 模型架构

模型 (`FusionModel` in `model.py`) 采用多模态融合策略：

1.  **图像编码器**: 使用预训练的 ResNet-152 提取图像的全局特征和序列特征。
2.  **光谱编码器**: 使用一系列 1D 卷积层处理光谱数据，提取全局特征和序列特征。
3.  **融合模块**: 将图像和光谱的序列特征与一个可学习的 `[CLS]` token 拼接，并添加 token 类型嵌入和位置嵌入。
4.  **Transformer Encoder**: 使用标准的 PyTorch Transformer Encoder 处理融合后的序列。
5.  **分类头**: 使用 `[CLS]` token 的 Transformer 输出进行最终的分类。
6.  **对比损失**: 同时计算图像和光谱全局特征之间的对比损失（InfoNCE 形式），与分类损失结合，以促进模态对齐。

## 训练与评估

执行以下命令开始训练：

```bash
python train.py
```

脚本将：

1.  加载数据并进行预处理。
2.  分割训练集和验证集。
3.  初始化模型、优化器和损失函数。
4.  执行训练循环，并在每个轮次后进行验证。
5.  打印每轮的损失、准确率、精确率、召回率和 F1 分数。
6.  将验证 F1 分数最高的模型保存到 `checkpoints/` 目录下（文件名包含训练 ID）。
7.  训练结束后，加载最佳模型在验证集上进行最终评估，并打印详细的分类报告。
8.  将本次训练的配置、逐轮指标、最终结果和分类报告追加到 `training_history.json` 文件中。
9.  生成包含训练过程指标（损失、准确率、精确率、召回率、F1）的 PNG 图表，保存到 `plots/` 目录下（文件名包含训练 ID）。

## 输出

*   **模型**: 训练好的模型检查点保存在 `checkpoints/` 目录。
*   **历史记录**: 详细的训练历史保存在 `training_history.json` 文件中。
*   **图表**: 训练过程的可视化图表保存在 `plots/` 目录。

## 后端 API 服务 (Backend API Service)

本项目包含一个基于 FastAPI 的后端 API 服务，定义在 `app.py` 文件中。该服务负责：

*   提供恒星数据 (从 `objects.csv` 和关联文件)。
*   解析光谱文件。
*   使用训练好的模型进行分类预测。
*   为前端提供静态文件服务 (图像和光谱数据)。

**运行后端服务 (基本步骤):**

1.  **确保依赖已安装**: 参考上面的 "## 依赖项" 部分安装必要的 Python 包，特别是 `fastapi` 和 `uvicorn`。如果尚未安装，可以使用：
    ```bash
    pip install fastapi "uvicorn[standard]" pandas Pillow astropy
    ```
    *(请确保所有在 `app.py` 中导入的库都已安装)*

2.  **启动服务**: 在项目根目录下，使用 `uvicorn` 运行 FastAPI 应用。
    ```bash
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `app:app` 指的是 `app.py` 文件中的 `app` FastAPI 实例。
    *   `--reload` 会在代码更改时自动重启服务，这在开发时非常方便。
    *   `--host 0.0.0.0` 使服务可以从本地网络中的其他设备访问（如果防火墙允许）。如果只想本机访问，可以使用 `127.0.0.1`。
    *   `--port 8000` 指定服务监听的端口。

服务启动后，API 通常可在 `http://localhost:8000` (或 `http://127.0.0.1:8000`) 访问。前端应用会默认连接此地址。您可以在浏览器中访问 `http://localhost:8000/docs` 查看自动生成的 API 文档。

## 前端应用 (Frontend Application)

本项目包含一个基于 Vue.js 和 Quasar框架的前端应用，位于 `mstar-classification-frontend/` 目录下。该前端应用旨在提供一个用户友好的界面，用于与后端 FastAPI 服务进行交互，实现以下功能：

*   **数据管理**: 浏览 `objects.csv` 中的恒星数据，查看关联的光谱和图像文件信息 (通过 `DataManagerPage.vue`)。
*   **数据可视化**:
    *   查看指定恒星的图像预览。
    *   (功能规划中) 跳转到光谱和图像的详细可视化页面。
*   **分类预测**: (功能规划中) 选择模型、光谱和图像文件，进行在线分类预测。
*   **导航**: 通过侧边栏在不同功能页面间切换 (如 `MainLayout.vue` 和 `router/routes.ts` 中所定义)。

前端通过 HTTP 请求与运行在 `http://localhost:8000` 的后端 API (`app.py`) 通信，以获取数据列表、文件内容和执行预测。

**运行前端 (基本步骤):**

```bash
cd mstar-classification-frontend
# 安装依赖 (如果需要)
# npm install  # 或者 yarn install
# 启动开发服务器
# quasar dev   # 或者 npm run dev / yarn dev
```
*具体的启动命令可能因项目配置而异。请查阅 `mstar-classification-frontend/package.json` 中的脚本命令。*

