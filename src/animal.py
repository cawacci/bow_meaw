# import torchvision and torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms

# import efficientnet
from efficientnet_pytorch import EfficientNet

# データ変換の定義
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.feature = EfficientNet.from_pretrained('efficientnet-b1')
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return(h)