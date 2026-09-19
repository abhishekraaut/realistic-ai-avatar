import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class MotionDiffusionUnet(nn.Module):
    """
    Skeletal Motion Diffusion Model converting Audio RVQ tokens to 3D spatial coordinates.
    """
    def __init__(self, audio_dim=512, motion_dim=156, hidden_dim=256):
        # motion_dim 156 could represent SMPL-X (52 joints * 3 coordinates)
        super().__init__()
        self.time_embed = TimestepEmbedder(hidden_dim)
        
        self.proj_in = nn.Linear(motion_dim, hidden_dim)
        
        # Cross attention for audio conditioning
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # Simple U-Net style transformer layers for sequence processing
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        
        self.proj_out = nn.Linear(hidden_dim, motion_dim)
        
    def forward(self, motion_x, t, audio_cond):
        """
        motion_x: (B, T, motion_dim) Noisy motion sequence
        t: (B,) Timesteps
        audio_cond: (B, T, audio_dim) Audio context
        """
        t_emb = self.time_embed(t).unsqueeze(1) # (B, 1, hidden_dim)
        
        x = self.proj_in(motion_x) # (B, T, hidden_dim)
        x = x + t_emb
        
        a_cond = self.audio_proj(audio_cond) # (B, T, hidden_dim)
        
        # Combine motion and audio conditioning via simple addition (or cross-attention in full model)
        x = x + a_cond
        
        x = self.transformer(x)
        
        out = self.proj_out(x) # (B, T, motion_dim)
        return out

class ExpressAnimateEngine:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = MotionDiffusionUnet().to(device)
        self.num_timesteps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.num_timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def predict_motion(self, audio_tokens):
        # audio_tokens: (B, T, audio_dim)
        B, T, _ = audio_tokens.shape
        audio_tokens = audio_tokens.to(self.device)
        
        # Start from pure noise
        x = torch.randn(B, T, 156, device=self.device)
        
        # Reverse diffusion loop
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = self.model(x, t, audio_tokens)
            
            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            
        return x
