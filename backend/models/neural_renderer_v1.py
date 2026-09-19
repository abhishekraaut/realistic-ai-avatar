import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 512x512 -> 256x256
        self.down1 = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU())
        # 256x256 -> 128x128
        self.down2 = nn.Sequential(nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        # 128x128 -> 64x64
        self.down3 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        # 64x64 -> 32x32
        self.down4 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        # 32x32 -> 16x16
        self.down5 = nn.Sequential(nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.res_blocks(x)
        return x

class MotionConditioning(nn.Module):
    def __init__(self, motion_dim=11, latent_channels=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(motion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_channels * 2) # Affine params (gamma, beta) for FiLM
        )

    def forward(self, identity_latent, motion):
        # identity_latent: [B, C, H, W]
        # motion: [B, motion_dim]
        affine_params = self.mlp(motion) # [B, C*2]
        gamma, beta = affine_params.chunk(2, dim=1)
        
        gamma = gamma.unsqueeze(2).unsqueeze(3) # [B, C, 1, 1]
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        return identity_latent * (1 + gamma) + beta

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(2)])
        
        # 16x16 -> 32x32
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        # 32x32 -> 64x64
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        # 64x64 -> 128x128
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
        # 128x128 -> 256x256
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU())
        # 256x256 -> 512x512
        self.up5 = nn.Sequential(nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x

class NeuralRendererV1(nn.Module):
    """
    Phase 8B: Static Image Reconstruction Proof-of-Concept
    Validates whether the 11D PCA Motion Latent can drive facial articulation.
    """
    def __init__(self, motion_dim=11):
        super().__init__()
        self.identity_encoder = IdentityEncoder()
        self.conditioner = MotionConditioning(motion_dim=motion_dim, latent_channels=128)
        self.decoder = Decoder()

    def forward(self, identity_img, motion_vector):
        """
        identity_img: [B, 3, 512, 512] (Reference frame)
        motion_vector: [B, 11] (Pose + PCA latents)
        """
        ident_latent = self.identity_encoder(identity_img)
        cond_latent = self.conditioner(ident_latent, motion_vector)
        out_img = self.decoder(cond_latent)
        return out_img
