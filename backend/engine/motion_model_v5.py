import torch
import torch.nn as nn
import os

class MotionModelV5(nn.Module):
    """
    Phase 8D: Identity-Conditioned Expressive Motion Predictor.
    Resolves V4 out-of-distribution PCA amplitude damping by conditioning
    the audio-to-motion mapping on a pooled identity representation.
    """
    def __init__(self, audio_dim=1024, id_dim=256, motion_dim=11):
        super().__init__()
        self.id_proj = nn.Linear(id_dim, 256)
        self.audio_proj = nn.Linear(audio_dim, 512)
        
        self.mlp = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, motion_dim)
        )
        
    def forward(self, audio_feat, id_feat):
        """
        audio_feat: [B, 1024] from Wav2Vec2/HuBERT
        id_feat: [B, 256] pooled from V3 Identity Encoder bottleneck (f4)
        """
        a = torch.relu(self.audio_proj(audio_feat))
        i = torch.relu(self.id_proj(id_feat))
        x = torch.cat([a, i], dim=-1)
        return self.mlp(x)

if __name__ == '__main__':
    model = MotionModelV5()
    print(f"V5 Parameters: {sum(p.numel() for p in model.parameters())}")
    audio = torch.randn(4, 1024)
    id_f = torch.randn(4, 256)
    out = model(audio, id_f)
    print(f"Output shape: {out.shape}")
