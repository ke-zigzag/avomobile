import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.quantization

#densenetの特徴量をインポート
from torchvision.models import mobilenet_v3_small
feature =  mobilenet_v3_small()

#モデルの定義
class Net(pl.LightningModule):

    def __init__(self):
       super().__init__()
       self.feature = mobilenet_v3_small()
       self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

#モデルインスタンスを作成し、重みを読み込む
avomodel = Net().cpu().eval()
avomodel.load_state_dict(torch.load('mobilenet.pt', map_location=torch.device('cpu')))


#モデルの量子化
#dummy_input = torch.randn(1, 3, 224, 224)  # 仮の入力サイズを指定
# = torch.quantization.quantize_dynamic(
    #model, {nn.Linear}, dtype=torch.qint8
#)