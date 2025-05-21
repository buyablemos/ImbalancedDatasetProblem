from torchvision import models
import torch.nn as nn

# 1) Define a pure PyTorch module
class ResNetModule(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet34'):
        super().__init__()
        self.net = getattr(models, model_name)(weights='IMAGENET1K_V1')
        num_features = self.net.fc.in_features
        self.net.fc = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.net(x)