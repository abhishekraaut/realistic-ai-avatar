import torch
import torch.nn as nn

class LearnedAudioEncoder(nn.Module):
    """
    Learns a temporal audio representation from streaming features (e.g., log-mel).
    Input: (B, T, C) where C=80 (log-mel bins)
    Output: (B, T, hidden_dim)
    """
    def __init__(self, input_dim=80, hidden_dim=64):
        super().__init__()
        # Simple 1D convolution to capture local context (e.g. phonemes)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x is (B, T, input_dim)
        x = x.transpose(1, 2) # (B, input_dim, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2) # (B, T, hidden_dim)
        return x

class LearnedFacialMotionDecoder(nn.Module):
    """
    Decodes the temporal audio representation into continuous facial motion.
    Input: (B, T, hidden_dim)
    Output: (B, T, output_dim)
    """
    def __init__(self, hidden_dim=64, output_dim=4):
        super().__init__()
        # LSTM for temporal continuity (e.g., mouth opening/closing smoothly)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class LearnedTemporalMotionModel(nn.Module):
    """
    End-to-End model: Audio -> Temporal Embedding -> Facial Motion
    """
    def __init__(self, input_dim=80, hidden_dim=64, output_dim=4):
        super().__init__()
        self.encoder = LearnedAudioEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = LearnedFacialMotionDecoder(hidden_dim=hidden_dim, output_dim=output_dim)
        
    def forward(self, x):
        features = self.encoder(x)
        motion = self.decoder(features)
        return motion
