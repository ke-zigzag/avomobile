#必要なライブラリーのインストール
from fastapi import FastAPI, File, UploadFile
import io
from model import avomodel
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm #背景切り取りライブラリー
from rembg import remove
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()

#画像変換の定義
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 画像サイズの変更
    transforms.ToTensor(),           # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化
])

@app.get('/')
async def index():
    return {"avocado": 'avocado_checker'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #画像を読み込む
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    #画像をモデルに適した形に変換
    output = remove(image).convert('RGB')
    image_tensor = transform(output)
    
    #予測を実行
    prediction = avomodel(image_tensor.unsqueeze(0))
    y = F.softmax(prediction, dim=1)
    result = torch.argmax(y, dim=1).item()
    result = str(result)
    
    return JSONResponse(content=jsonable_encoder({"result": result}))
    