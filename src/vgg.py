import torch.nn as nn
from torchvision import models

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg_features = models.vgg16(pretrained=True).features
        self._slice_1 = nn.Sequential()
        self._slice_2 = nn.Sequential()
        self._slice_3 = nn.Sequential()
        self._slice_4 = nn.Sequential()
        for x in range(4):
            self._slice_1.add_module(str(x), vgg_features[x])
        for x in range(4, 9):
            self._slice_2.add_module(str(x), vgg_features[x])
        for x in range(9, 16):
            self._slice_3.add_module(str(x), vgg_features[x])
        for x in range(16, 23):
            self._slice_4.add_module(str(x), vgg_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self._slice_1(x)
        relu1_2 = x
        x = self._slice_2(x)
        relu2_2 = x
        x = self._slice_3(x)
        relu3_3 = x
        x = self._slice_4(x)
        relu4_3 = x
        out = {'relu1_2': relu1_2,
               'relu2_2': relu2_2,
               'relu3_3': relu3_3,
               'relu4_3': relu4_3}
        return out