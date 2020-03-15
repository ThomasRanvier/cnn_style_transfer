import torch.nn as nn

class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()
        ## (3, 256, 256) ## padding = (left, right, top, bottom)
        self._conv_1 = GenericLayer(nn.Conv2d(3, 32, 9, 1), 32, (5, 5, 5, 5), nn.ReLU())
        ## (32, 256, 256)
        self._conv_2 = GenericLayer(nn.Conv2d(32, 64, 3, 2), 64, (1, 0, 1, 0), nn.ReLU())
        ## (64, 128, 128)
        self._conv_3 = GenericLayer(nn.Conv2d(64, 128, 3, 2), 128, (1, 0, 1, 0), nn.ReLU())
        ## (128, 64, 64)
        self._res_1 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        ## (128, 64, 64)
        self._res_2 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        ## (128, 64, 64)
        self._res_3 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        ## (128, 64, 64)
        self._res_4 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        ## (128, 64, 64)
        self._res_5 = ResidualBlock(128, 3, 1, (1, 1, 1, 1))
        ## (128, 64, 64)
        self._conv_4 = GenericLayer(nn.ConvTranspose2d(128, 64, 3, 2, output_padding=1), 64, (-1, 0, -1, 0), nn.ReLU())
        ## (64, 128, 128)
        self._conv_5 = GenericLayer(nn.ConvTranspose2d(64, 32, 3, 2, output_padding=1), 32, (-1, 0, -1, 0), nn.ReLU())
        ## (32, 256, 256)
        self._conv_6 = GenericLayer(nn.Conv2d(32, 3, 9, 1), 3, (4, 4, 4, 4), nn.Sigmoid())
        ## (3, 256, 256)
        
    def forward(self, x):
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = self._res_1(x)
        x = self._res_2(x)
        x = self._res_3(x)
        x = self._res_4(x)
        x = self._res_5(x)
        x = self._conv_4(x)
        x = self._conv_5(x)
        x = self._conv_6(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding=(0,0,0,0)):
        super(ResidualBlock, self).__init__()
        self._conv_1 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 1, 1), nn.ReLU())
        self._conv_2 = GenericLayer(nn.Conv2d(128, 128, 3, 1), 128, (1, 1, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self._conv_1(x)
        x = x + self._conv_2(x)
        return x

class GenericLayer(nn.Module):
    def __init__(self, layer, out_channels, padding=(0,0,0,0), activation=None):
        super(GenericLayer, self).__init__()
        self._act = activation
        self._layer = layer
        self._norm = nn.BatchNorm2d(out_channels)
        self._pad = nn.ZeroPad2d(padding)

    def forward(self, x):
        x = self._pad(x)
        x = self._layer(x)
        x = self._norm(x)
        if self._act is not None:
            x = self._act(x)
        return x