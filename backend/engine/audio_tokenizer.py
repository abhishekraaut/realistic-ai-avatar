import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualVectorQuantizer(nn.Module):
    """
    Custom RVQ layer to compress continuous audio embeddings into discrete tokens.
    """
    def __init__(self, num_quantizers=8, codebook_size=1024, dim=512):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim)) for _ in range(num_quantizers)
        ])

    def forward(self, x):
        """
        x: (B, T, D) continuous audio embeddings
        Returns: quantized (B, T, D), tokens (B, T, num_quantizers)
        """
        B, T, D = x.shape
        quantized_out = torch.zeros_like(x)
        residual = x
        all_tokens = []
        
        for i in range(self.num_quantizers):
            cb = self.codebooks[i] # (V, D)
            # Compute distances: (B, T, V)
            dist = torch.cdist(residual.view(-1, D), cb).view(B, T, self.codebook_size)
            tokens = torch.argmin(dist, dim=-1) # (B, T)
            all_tokens.append(tokens)
            
            # Lookup selected codes
            quantized = F.embedding(tokens, cb)
            
            # Straight-through estimator trick
            quantized = residual + (quantized - residual).detach()
            
            quantized_out = quantized_out + quantized
            residual = residual - quantized
            
        return quantized_out, torch.stack(all_tokens, dim=-1)

class AudioTransformerDecoder(nn.Module):
    """
    Autoregressive Transformer predicting RVQ tokens from Text embeddings.
    """
    def __init__(self, vocab_size=50000, dim=512, depth=12, heads=8):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(4096, dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        self.to_logits = nn.Linear(dim, 1024) # Predicts 1st level RVQ codebook

    def forward(self, text_seq, audio_tokens=None):
        B, T_text = text_seq.shape
        text_features = self.text_emb(text_seq)
        
        if audio_tokens is None:
            # Inference mode logic would autoregressively sample here
            # For simplicity, returning a dummy tensor of sequence length 1
            audio_seq = torch.zeros(B, 1, self.text_emb.embedding_dim, device=text_seq.device)
        else:
            # Training mode: (B, T_audio, dim) 
            audio_seq = audio_tokens 
            
        pos = torch.arange(audio_seq.shape[1], device=text_seq.device).unsqueeze(0)
        audio_seq = audio_seq + self.pos_emb(pos)
        
        # Causal mask for audio sequence
        T_audio = audio_seq.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T_audio).to(text_seq.device)
        
        out = self.transformer(audio_seq, text_features, tgt_mask=causal_mask)
        logits = self.to_logits(out)
        return logits

class ExpressVoiceEngine(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.rvq = ResidualVectorQuantizer().to(device)
        self.transformer = AudioTransformerDecoder().to(device)
        self.audio_decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(128, 1, kernel_size=4, stride=2, padding=1)
        ).to(device)
        
    def generate_audio(self, text_tokens):
        # Predict tokens autoregressively (mocked loop)
        text_tokens = text_tokens.to(self.device)
        # B, T_audio, V -> In production, we loop and sample from logits
        dummy_audio_features = torch.randn(text_tokens.size(0), 100, 512, device=self.device)
        
        # Decode back to audio waveform
        dummy_audio_features = dummy_audio_features.transpose(1, 2) # (B, D, T)
        audio_waveform = self.audio_decoder(dummy_audio_features)
        return audio_waveform
