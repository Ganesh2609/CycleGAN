import torch
from torch import nn


class Discriminator(nn.Module):
    
    def __init__(self, in_channels:int, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        layers = nn.ModuleList()
        layers.append(self.conv_block(in_channels=in_channels, out_channels=features[0], kernel_size=4, stride=2, padding=1, norm=False))
        prev_channels = features[0]
        for curr_channels in features[1:-1]:
            layers.append(self.conv_block(in_channels=prev_channels, out_channels=curr_channels, kernel_size=4, stride=2, padding=1, norm=True))
            prev_channels = curr_channels
        layers.append(self.conv_block(in_channels=prev_channels, out_channels=features[-1], kernel_size=4, stride=1, padding=1, norm=True))
        layers.append(nn.Conv2d(in_channels=features[-1], out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect', bias=True))
        self.model = nn.Sequential(*layers)
    
    def conv_block(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, norm:bool):
        if norm:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=True),
                nn.InstanceNorm2d(num_features=out_channels, affine=True),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect', bias=True),
                nn.LeakyReLU(negative_slope=0.2)
            )
        
    def forward(self, x):
        return self.model(x)