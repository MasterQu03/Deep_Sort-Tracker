import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class ResNet50(nn.Module):
    def __init__(self, sex_classes, age_classes, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.sex_classifier = nn.Linear(2048, sex_classes)
        self.age_classifier = nn.Linear(2048, age_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # if not self.training:
        #     return f
        sex = self.sex_classifier(f)
        age = self.age_classifier(f)
        return sex, age
