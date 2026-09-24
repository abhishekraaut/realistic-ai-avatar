import torch
import torch.nn as nn
import torch.optim as optim
import os
import hashlib
import time

class ContextualAudioEncoder(nn.Module):
    def __init__(self, input_dim=80, context_size=16, embed_dim=64):
        super().__init__()
        self.context_size = context_size
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        pooled_len = context_size // 2
        self.fc = nn.Linear(64 * pooled_len, embed_dim)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc(x))
        return x

class LearnedTemporalMotionModelV5(nn.Module):
    def __init__(self, input_dim=80, context_size=16, hidden_dim=64, output_dim=52):
        super().__init__()
        self.audio_encoder = ContextualAudioEncoder(input_dim, context_size, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        B, S, C, F_dim = x.size()
        x_flat = x.view(B * S, C, F_dim)
        emb = self.audio_encoder(x_flat)
        emb_seq = emb.view(B, S, -1)
        lstm_out, _ = self.lstm(emb_seq)
        out = self.decoder(lstm_out)
        return out


def train():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = torch.load('synthesia_training_data/dataset_v5.pt', map_location='cpu')
    X_train = dataset['train']['X']
    Y_train = dataset['train']['Y']
    
    # Make sequences
    batch_size = 4
    seq_len = 32
    
    X_seq = []
    Y_seq = []
    for i in range(0, X_train.shape[0] - seq_len, seq_len):
        X_seq.append(X_train[i:i+seq_len])
        Y_seq.append(Y_train[i:i+seq_len])
    
    if not X_seq:
        X_seq.append(X_train[:seq_len])
        Y_seq.append(Y_train[:seq_len])
        
    X_batch = torch.stack(X_seq[:batch_size]).to(device)
    Y_batch = torch.stack(Y_seq[:batch_size]).to(device)
    
    model = LearnedTemporalMotionModelV5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    initial_loss = None
    final_loss = None
    
    t0 = time.time()
    for step in range(50):
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, Y_batch)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
    t1 = time.time()
    
    # Validation (dummy batch just to check forward works)
    with torch.no_grad():
        val_out = model(X_batch)
        val_loss = criterion(val_out, Y_batch).item()
    
    os.makedirs('backend/training/checkpoints', exist_ok=True)
    checkpoint_path = 'backend/training/checkpoints/motion_v5_smoke.pt'
    torch.save(model.state_dict(), checkpoint_path)
    
    file_size = os.path.getsize(checkpoint_path)
    sha256 = hashlib.sha256(open(checkpoint_path, 'rb').read()).hexdigest()
    num_params = sum(p.numel() for p in model.parameters())
    
    # Verify reload
    new_model = LearnedTemporalMotionModelV5().to(device)
    new_model.load_state_dict(torch.load(checkpoint_path))
    new_model.eval()
    with torch.no_grad():
        test_out = new_model(X_batch)
    
    print(f'Initial Loss: {initial_loss:.6f}')
    print(f'Final Loss: {final_loss:.6f}')
    print(f'Val Loss: {val_loss:.6f}')
    print(f'Runtime: {t1 - t0:.2f}s')
    print(f'Params: {num_params}')
    print(f'Size: {file_size}')
    print(f'SHA256: {sha256}')
    print(f'Inference OK: {test_out.shape == Y_batch.shape}')

if __name__ == '__main__':
    train()
