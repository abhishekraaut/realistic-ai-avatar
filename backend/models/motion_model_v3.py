import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualAudioEncoder(nn.Module):
    def __init__(self, input_dim=80, context_size=16, embed_dim=64):
        super().__init__()
        self.context_size = context_size
        
        # Convolutions over the context temporal dimension
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # After two max pools (if applied) or just one, context size reduces.
        # Let's just use one MaxPool1d(2). So context_size -> context_size // 2
        pooled_len = context_size // 2
        self.fc = nn.Linear(64 * pooled_len, embed_dim)
        
    def forward(self, x):
        # x: [B, Context, 80]
        x = x.transpose(1, 2) # [B, 80, Context]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)      # [B, 64, Context // 2]
        
        x = x.view(x.size(0), -1) # [B, 64 * (Context//2)]
        x = F.relu(self.fc(x))    # [B, embed_dim]
        return x

class LearnedTemporalMotionModelV3(nn.Module):
    def __init__(self, input_dim=80, context_size=16, hidden_dim=64, output_dim=4):
        super().__init__()
        
        self.audio_encoder = ContextualAudioEncoder(input_dim, context_size, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [Batch, Seq, Context, 80]
        B, S, C, F_dim = x.size()
        
        # Process each step's context independently
        x_flat = x.view(B * S, C, F_dim)
        emb = self.audio_encoder(x_flat) # [B * S, hidden_dim]
        
        # Reshape to sequence
        emb_seq = emb.view(B, S, -1)     # [B, S, hidden_dim]
        
        lstm_out, _ = self.lstm(emb_seq)
        
        out = self.decoder(lstm_out)     # [B, S, 4]
        return out
