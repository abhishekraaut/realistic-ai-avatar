import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, cond_dim, feature_dim):
        super().__init__()
        self.fc = nn.Linear(cond_dim, feature_dim * 2)
        
    def forward(self, x, cond):
        gamma_beta = self.fc(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return x * (1 + gamma) + beta

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = ConvBlock(3, 32, stride=2)   # 256
        self.b2 = ConvBlock(32, 64, stride=2)  # 128
        self.b3 = ConvBlock(64, 128, stride=2) # 64
        self.b4 = ConvBlock(128, 256, stride=2)# 32
        
    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        return f1, f2, f3, f4

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_c + skip_c, out_c)
        self.film = FiLM(cond_dim, out_c)
        
    def forward(self, x, skip, cond):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.film(x, cond)
        return x

class NeuralRendererV2(nn.Module):
    def __init__(self, motion_dim=11):
        super().__init__()
        self.encoder = IdentityEncoder()
        
        self.motion_mlp = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = DecoderBlock(256, 128, 128, cond_dim=256) # 32 -> 64
        self.dec3 = DecoderBlock(128, 64, 64, cond_dim=256)   # 64 -> 128
        self.dec2 = DecoderBlock(64, 32, 32, cond_dim=256)    # 128 -> 256
        self.dec1 = DecoderBlock(32, 0, 16, cond_dim=256)     # 256 -> 512
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, identity_ref, motion):
        # Identity Reference: [B, 3, H, W]
        # Motion: [B, 11]
        
        f1, f2, f3, f4 = self.encoder(identity_ref)
        cond = self.motion_mlp(motion)
        
        d = self.dec4(f4, f3, cond)
        d = self.dec3(d, f2, cond)
        d = self.dec2(d, f1, cond)
        d = self.dec1(d, None, cond)
        
        out = self.out_conv(d)
        return out
