import torch 
from torch import nn 



class ConvBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, down:bool, act:bool, norm:bool, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding_mode='reflect', **kwargs)
            if down else
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.InstanceNorm2d(num_features=out_channels, affine=True) if norm else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)
    


class ResidualBlock(nn.Module):
    
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            ConvBlock(in_channels=num_channels, out_channels=num_channels, down=True, act=True, norm=True, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=num_channels, out_channels=num_channels, down=True, act=False, norm=True, kernel_size=3, stride=1, padding=1)
        )
        
    def forward(self, x):
        return x + self.residual(x)



class Generator(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, num_features:int, num_residuals:int):
        super(Generator, self).__init__()
        self.initial_conv = ConvBlock(in_channels=in_channels, out_channels=num_features, down=True, act=True, norm=False, kernel_size=7, stride=1, padding=3)
        self.downs = nn.Sequential(
            ConvBlock(in_channels=num_features, out_channels=num_features*2, down=True, act=True, norm=True, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=num_features*2, out_channels=num_features*4, down=True, act=True, norm=True, kernel_size=3, stride=2, padding=1)
        )
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_channels=num_features*4) for _ in range(num_residuals)]
        )
        self.ups = nn.Sequential(
            ConvBlock(in_channels=num_features*4, out_channels=num_features*2, down=False, act=False, norm=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(in_channels=num_features*2, out_channels=num_features, down=False, act=False, norm=False, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_features, out_channels=out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.downs(x)
        x = self.residuals(x)
        x = self.ups(x)
        x = self.final_conv(x)
        return x