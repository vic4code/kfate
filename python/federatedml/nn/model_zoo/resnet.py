
# model
import torch as t
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class Resnet(nn.Module):

    def __init__(self, ):
        super(Resnet, self).__init__()
        self.resnet = resnet18()
        self.classifier = t.nn.Linear(1000, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.training:
            return self.classifier(self.resnet(x))
        else:
            return self.softmax(self.classifier(self.resnet(x)))
    
