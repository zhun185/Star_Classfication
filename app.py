from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import torch
from astropy.io import fits
from PIL import Image
import numpy as np
from pathlib import Path
import csv # 导入 csv 模块
import traceback # 新增导入

# 尝试导入 pandas，如果失败则标记，以便后续优雅处理
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# 从 model.py 导入 FusionModel 类
from model import FusionModel 

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
SPECTRA_DIR = BASE_DIR / "spectra"
IMAGES_DIR = BASE_DIR / "images"
CSV_FILE_PATH = BASE_DIR / "Skyserver_Radial5_9_2025 5_12_34 AM.csv"

# 定义允许的源列表
origins = [
    "http://localhost:8080",  # Quasar 前端开发服务器地址
    # 如果您有生产环境的前端地址，也需要加到这里
    # "https://your.production-frontend.com",
]

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 新增：挂载静态文件目录
# 前端可以通过 /static/images/<subclass>/<image_file_name> 访问图像
# 前端可以通过 /static/spectra/<subclass>/<spectrum_file_name> 访问光谱文件
if IMAGES_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="static_images")
if SPECTRA_DIR.exists():
    app.mount("/static/spectra", StaticFiles(directory=SPECTRA_DIR), name="static_spectra")

# 确保目录存在，如果不存在则创建
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(SPECTRA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# 获取模型列表
@app.get("/api/models")
def get_models():
    models = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith(".pth")]
    return models

# 新增：列出可用的光谱文件
@app.get("/api/available_spectra")
def list_available_spectra():
    available_files = []
    for root, _, files in os.walk(SPECTRA_DIR):
        for file in files:
            if file.lower().endswith(('.fit', '.fits')):
                # 保存相对于 SPECTRA_DIR 的路径
                relative_path = Path(root).relative_to(SPECTRA_DIR) / file
                available_files.append(str(relative_path.as_posix())) # 使用 POSIX 路径风格
    return sorted(available_files)

# 新增：列出可用的图像文件
@app.get("/api/available_images")
def list_available_images():
    available_files = []
    for root, _, files in os.walk(IMAGES_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 保存相对于 IMAGES_DIR 的路径
                relative_path = Path(root).relative_to(IMAGES_DIR) / file
                available_files.append(str(relative_path.as_posix())) # 使用 POSIX 路径风格
    return sorted(available_files)

# 新增：解析 CSV 并关联文件，提供结构化数据
@app.get("/api/star_data")
def get_star_data():
    star_data_list = []
    if not CSV_FILE_PATH.exists():
        return JSONResponse(status_code=404, content={"detail": "Skyserver CSV file not found."})

    # 定义预期的图像扩展名列表
    image_extensions = ['.jpg', '.jpeg', '.png']

    if PANDAS_AVAILABLE:
        try:
            df = pd.read_csv(CSV_FILE_PATH, comment='#') # 跳过以 # 开头的注释行
            # 清理列名，去除可能存在的前后空格
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                specobjid = str(row['specobjid']).strip()
                subclass = str(row['subclass']).strip()
                
                spectrum_filename = f"spec-{specobjid}.fits"
                spectrum_relative_path = Path(subclass) / spectrum_filename
                full_spectrum_path = SPECTRA_DIR / spectrum_relative_path

                # 尝试所有可能的图像扩展名
                image_found = False
                image_relative_path_str = ""
                for ext in image_extensions:
                    image_filename = f"img-{specobjid}{ext}"
                    image_relative_path = Path(subclass) / image_filename
                    full_image_path = IMAGES_DIR / image_relative_path
                    if full_image_path.exists() and full_image_path.is_file():
                        image_relative_path_str = str(image_relative_path.as_posix())
                        image_found = True
                        break
                
                if full_spectrum_path.exists() and full_spectrum_path.is_file() and image_found:
                    star_info = row.to_dict()
                    star_info['spectrum_path'] = str(spectrum_relative_path.as_posix())
                    star_info['image_path'] = image_relative_path_str
                    star_data_list.append(star_info)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Error processing CSV with pandas: {str(e)}"})
    else: # 使用 csv 模块
        try:
            with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
                # 跳过注释行并读取表头
                reader = None
                header = []
                for line in csvfile:
                    if line.startswith('#'):
                        continue
                    header = [h.strip() for h in line.strip().split(',')]
                    reader = csv.DictReader(csvfile, fieldnames=header) # 从下一行开始作为数据
                    break # 已找到表头并初始化 DictReader
                
                if reader is None:
                     return JSONResponse(status_code=500, content={"detail": "CSV file seems empty or only contains comments."})

                for row in reader:
                    specobjid = str(row['specobjid']).strip()
                    subclass = str(row['subclass']).strip()

                    spectrum_filename = f"spec-{specobjid}.fits"
                    spectrum_relative_path = Path(subclass) / spectrum_filename
                    full_spectrum_path = SPECTRA_DIR / spectrum_relative_path

                    image_found = False
                    image_relative_path_str = ""
                    for ext in image_extensions:
                        image_filename = f"img-{specobjid}{ext}"
                        image_relative_path = Path(subclass) / image_filename
                        full_image_path = IMAGES_DIR / image_relative_path
                        if full_image_path.exists() and full_image_path.is_file():
                            image_relative_path_str = str(image_relative_path.as_posix())
                            image_found = True
                            break
                    
                    if full_spectrum_path.exists() and full_spectrum_path.is_file() and image_found:
                        star_info = {k.strip(): v.strip() for k, v in row.items()} # 清理键和值
                        star_info['spectrum_path'] = str(spectrum_relative_path.as_posix())
                        star_info['image_path'] = image_relative_path_str
                        star_data_list.append(star_info)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Error processing CSV with csv module: {str(e)}"})
            
    return star_data_list

# 解析FITS文件，返回光谱数据
# 修改：使其也能通过文件路径读取光谱 (可选，根据前端需求决定是否使用)
@app.post("/api/parse_spectrum")
async def parse_spectrum(spectrum_file: UploadFile = File(None), spectrum_path: str = Form(None)):
    file_to_process = None
    cleanup_file = False

    if spectrum_path:
        full_path = SPECTRA_DIR / spectrum_path
        if not full_path.exists() or not full_path.is_file():
            return JSONResponse(status_code=404, content={"detail": "Spectrum file not found at specified path."})
        file_to_process = full_path
    elif spectrum_file:
        # 为了能用 astropy.io.fits 打开 UploadFile，我们可能需要将其保存为临时文件
        # 或者直接使用 spectrum_file.file (file-like object)
        file_to_process = spectrum_file.file 
    else:
        return JSONResponse(status_code=400, content={"detail": "Either spectrum_file or spectrum_path must be provided."})

    try:
        with fits.open(file_to_process) as hdul:
            data = hdul[1].data
            flux = data['flux'].tolist() if 'flux' in data.columns.names else data.field(0).tolist()
            if 'wavelength' in data.columns.names:
                wavelength = data['wavelength'].tolist()
            elif 'loglam' in data.columns.names:
                wavelength = (10 ** data['loglam']).tolist()
            else:
                wavelength = list(range(len(flux)))
        return {"wavelength": wavelength, "flux": flux}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Error parsing FITS file: {str(e)}"})
    finally:
        # 如果创建了临时文件，这里需要清理
        pass 

# 预测API
# 修改：使其能接受光谱和图像的文件路径
@app.post("/api/predict")
async def predict(
    spectrum_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    model_name: str = Form(...)
):
    print(f"--- New Prediction Request ---")
    print(f"Received spectrum_file: {spectrum_file.filename if spectrum_file else 'None'}")
    print(f"Received image_file: {image_file.filename if image_file else 'None'}")
    print(f"Received model_name: {model_name}")

    spectrum_data_source = None
    image_data_source = None

    if not spectrum_file or not spectrum_file.filename:
        return JSONResponse(status_code=400, content={"detail": "Spectrum file is required and must have a filename."})
    spectrum_data_source = spectrum_file.file

    if not image_file or not image_file.filename:
        return JSONResponse(status_code=400, content={"detail": "Image file is required and must have a filename."})
    image_data_source = image_file.file
    
    try:
        print(f"Attempting to load model: {CHECKPOINTS_DIR / model_name}")
        model_path = CHECKPOINTS_DIR / model_name
        if not model_path.exists():
             print(f"Model file {model_path} does not exist on server.")
             return JSONResponse(status_code=404, content={"detail": f"Model file not found on server: {model_name}"})
        
        # 恢复实际模型加载和实例化
        print("Loading model state dict...")
        state_dict = torch.load(model_path, map_location='cpu')
        model_instance = FusionModel() # 确保 FusionModel 是您正确的模型类
        print("Loading state dict into model...")
        model_instance.load_state_dict(state_dict)
        model = model_instance
        model.eval()
        print("Model loaded successfully.")

        print("Processing spectrum data...")
        with fits.open(spectrum_data_source) as hdul:
            data = hdul[1].data
            flux_data = data['flux'] if 'flux' in data.columns.names else data.field(0)
            flux_data = np.array(flux_data, dtype=np.float32)
            flux_mean = np.mean(flux_data)
            flux_std = np.std(flux_data)
            flux_normalized = (flux_data - flux_mean) / (flux_std + 1e-8)
            flux_tensor = torch.tensor(flux_normalized).unsqueeze(0)
        print(f"Spectrum data processed. Shape: {list(flux_tensor.shape)}")

        print("Processing image data...")
        img = Image.open(image_data_source).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        print(f"Image data processed. Shape: {list(img_tensor.shape)}")

        # 恢复实际推理
        print("Performing inference...")
        with torch.no_grad():
            align_logits, fuse_logits = model(flux_tensor, img_tensor) # 确保 model() 调用是正确的
            prob = torch.softmax(fuse_logits, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred].item() * 100
        class_map = ['M0', 'M1', 'M2', 'M3', 'M4'] # 确保这个 class_map 与您的模型输出一致
        pred_class = class_map[pred] if pred < len(class_map) else str(pred)
        print(f"Prediction: {pred_class}, Confidence: {confidence:.2f}%")
        
        # 返回真实的预测结果
        return JSONResponse({"class": pred_class, "confidence": round(confidence, 2)})

    except Exception as e:
        error_details = f"Error during prediction processing in /api/predict: {str(e)}\n{traceback.format_exc()}"
        print(error_details) 
        return JSONResponse(status_code=500, content={"detail": error_details}) 

# 新增：通过API端点获取文件，确保CORS策略被应用
@app.get("/api/fetch_file/{file_type}/{sub_path:path}")
async def fetch_file_for_prediction(
    file_type: str, # FastAPI 会从路径中提取
    sub_path: str   # FastAPI 会从路径中提取
):
    print(f"API Call: /api/fetch_file/{file_type}/{sub_path}")
    base_dir_to_use = None
    if file_type == "spectrum":
        base_dir_to_use = SPECTRA_DIR
    elif file_type == "image":
        base_dir_to_use = IMAGES_DIR
    else:
        print(f"Invalid file_type: {file_type}")
        return JSONResponse(status_code=400, content={"detail": "Invalid file_type. Must be 'spectrum' or 'image'"})

    # Path 安全性: 确保 sub_path 不会逃逸出 base_dir_to_use
    # os.path.abspath 将解析路径，包括 ..
    # 然后我们检查它是否仍在预期的父目录下
    prospective_path = base_dir_to_use / sub_path
    # 使用 resolve() 来规范化路径 (处理 ../, ./ 等)
    full_file_path = prospective_path.resolve()

    # 检查解析后的路径是否仍在预期的父目录下
    if base_dir_to_use.resolve() not in full_file_path.parents and full_file_path != base_dir_to_use.resolve():
        # 如果文件与基本目录相同（例如 sub_path 为空或'.'），也允许，但通常 sub_path 不会是这样
        # 主要防止目录遍历攻击，如 sub_path = ../../etc/passwd
        if not (str(full_file_path).startswith(str(base_dir_to_use.resolve()))):
             print(f"Path traversal attempt or invalid path: {sub_path} resolved to {full_file_path}")
             return JSONResponse(status_code=400, content={"detail": "Invalid file path."})

    if not full_file_path.exists() or not full_file_path.is_file():
        print(f"File not found at: {full_file_path}")
        return JSONResponse(status_code=404, content={"detail": f"File not found: {sub_path}"})
    
    file_name_for_response = full_file_path.name
    print(f"Serving file: {full_file_path} as {file_name_for_response}")
    return FileResponse(path=str(full_file_path), filename=file_name_for_response) 