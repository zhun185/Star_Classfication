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

## J-PLUS DR3 图像数据下载脚本开发日志

用户最初询问如何将项目文件传输到远程服务器。我解释了可以使用 `scp` 命令，并询问了服务器的用户名、目标目录以及要传输的具体文件/目录，以便构建完整的 `scp` 命令。我提供了 `scp` 命令的基本格式。

接着，用户改变了主题，表示暂时不处理服务器问题，而是希望从 `https://archive.cefca.es/catalogues/` 下载数据，并使用 Pillow 库将 `i`, `r`, `g` 通道的图像合成为 RGB 图片。我指出该链接是一个数据门户，包含多个巡天项目，需要用户明确：
1.  感兴趣的具体巡天项目（例如 J-PLUS DR3）。
2.  如何确定下载目标（天体列表、坐标、ID 或特定天区）。
3.  如何关联目标天体的 `i`, `r`, `g` 波段图像文件。

用户确认希望使用 J-PLUS DR3 数据。我进一步询问了下载目标的指定方式（天体列表、坐标、ID 或区域标识符），J-PLUS 中对应 SDSS `i, r, g` 的滤光片名称，以及 J-PLUS DR3 的具体数据访问方式（网页查询、API、批量下载工具等）。我提到计划调研 J-PLUS DR3 的编程访问方式和滤光片信息。

我进行了两次网络搜索：
1.  第一次搜索 J-PLUS DR3 滤光片名称，结果指向一篇 A&A 的论文 "J-PLUS: photometric calibration of large-area multi-filter..." (López-Sanjuan et al., 2019)，其中提到了 J-PLUS 的12个滤光片包括标准的 `g(gSDSS)`, `r(rSDSS)`, `i(iSDSS)`。
2.  第二次搜索 J-PLUS DR3 编程数据访问或下载脚本，结果指向 CEFCA 的 J-PLUS DR3 数据访问服务页面，列出了 Sky Navigator, Object List Search, Image Search, Cone Search, V.O. Services (TAP, SIAP), ADQL 查询和 Direct Download Services。

基于搜索结果，我向用户确认了 J-PLUS DR3 中 `i, r, g` 波段的对应名称，并详细介绍了 J-PLUS DR3 的数据访问方式，特别是 V.O. 服务 (SIAP) 作为批量下载的潜力。我再次请用户提供下载目标的具体形式（如 CSV 文件及其包含的坐标或 ID）和本地存储目录结构。

用户随后提供了非常详细的数据要求，混合了 SDSS DR16 的光谱数据特性和 J-PLUS DR3 的图像数据需求：
1.  **光谱数据 (源自 SDSS DR16):**
    *   S/N 范围 0-94，分两组 (S/N < 5 和 S/N > 5)。
    *   波长范围 4000-9000 Å，3522 维向量（这是后期处理）。
2.  **光度数据/图像 (希望从 J-PLUS DR3 获取图像，但描述中部分引用 SDSS 特性):**
    *   M 型星 (M0-M4)。
    *   使用 J-PLUS 的 `i, r, g` 波段合成 RGB 图像 (i->R, r->G, g->B)。
    *   通过 RA, Dec 定位，获取 128x128 像素的裁剪图像。
    *   手动图像筛选条件（多源、偏离中心、噪声严重），这是后期处理。
3.  **恒星图像 (明确来自 J-PLUS DR3):**
    *   通过 `i, r, g` 合成，分辨率 128x128。

我分析了这些需求，指出光谱数据已从 SDSS 获取，核心新需求是从 J-PLUS DR3 获取与 SDSS M 型星匹配的 `i, r, g` 波段图像，裁剪并合成。我提出了使用 `objects.csv` (包含 SDSS M 型星坐标) 作为目标列表，通过 J-PLUS DR3 SIAP 服务下载图像的策略。我提到需要确认 J-PLUS DR3 SIAP 服务的具体细节（URL, BAND, POS, SIZE 参数）。

我再次进行了网络搜索，查找 J-PLUS DR3 SIAP 服务的具体参数。结果提供了 J-PLUS DR3 V.O. 服务页面链接，其中包含 SIAP 服务 URL (`https://archive.cefca.es/catalogues/vo/siap/jplus-dr3` 和 v2 的 `https://archive.cefca.es/catalogues/vo/siap2/jplus-dr3/query`)，并提到可以获取指定大小的 cutouts。我还从之前的论文信息中得知 J-PLUS 像素尺度约为 0.55 角秒/像素，用于估算 SIAP 查询的 `SIZE` 参数 (128 像素约 0.01955 度)。我指出 `BAND` 参数可能需要通过查询结果元数据筛选，并建议使用 `pyvo` 库。最后，我询问了用户 `objects.csv` 文件中 RA 和 Dec 列的具体名称。

用户提供了 `objects.csv` 的所有列名: `name,ra,dec,plate,mjd,fiberID,subclass,u_mag,g_mag,r_mag,i_mag,z_mag`。

基于此，我总结了最终目标：为 `objects.csv` 中的每个天体，从 J-PLUS DR3 下载 `g, r, i` 波段 FITS 图像，裁剪为 128x128 像素，然后用 Pillow 合成 RGB 图像 (i,r,g -> R,G,B)，并按 `subclass` 存放文件。我概述了详细的技术方案和 Python 脚本的主要步骤，涉及 `pandas`, `pyvo`, `astropy` (fits, wcs, Cutout2D), `Pillow`, `requests`, `os` 等库，并表示准备开始编写名为 `download_jplus_images.py` 的脚本。
