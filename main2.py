#必要なライブラリーのインストール
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from io import BytesIO
from model import avomodel
from PIL import Image
from torchvision import transforms
import torch
import torch.nn 
from torch.nn import functional as F
#背景切り取りライブラリー
#from tqdm import tqdm
from rembg import remove

app = FastAPI()

@app.get('/')
async def index():
    return {"avocado": 'avocado_checker'}

@app.get("/health")
async def health_check():
    return {"status": "OK"}


#画像の前処理
def preprocess(image: Image.Image):
    #背景切り取り
    output_image = remove(image).convert('RGB')
    output_image = np.asarray(output_image.resixe((224, 224)))[..., :3] #リサイズ
    output_image = np.expand_dims(output_image, 0) #バッチ次元の追加
    output_image = output_image / 255.0 #正規化
    return output_image

#テンソルへの変換の定義
transform = transforms.Compose([
    transforms.ToTensor(),           # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 標準化
])

def predict(image: Image.Image) -> str:
    preprocessed_image = preprocess(image)
    with torch.no_grad():
        prediction = avomodel(torch.from_numpy(preprocessed_image).float())
        y = torch.nn.functional.softmax(prediction, dim=1)
        result = torch.argmax(y, dim=1).item()

    return str(result)

@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    #画像を読み込む
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    result = predict(image)

    return JSONResponse(content=jsonable_encoder({"result":result}))
    
    