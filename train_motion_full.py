import torch
import torch.nn as nn
import torch.optim as optim
import os
import hashlib
import time
import numpy as np

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


def make_batches(X, Y, batch_size=4, seq_len=32):
    X_seq, Y_seq = [], []
    for i in range(0, X.shape[0] - seq_len, seq_len):
        X_seq.append(X[i:i+seq_len])
        Y_seq.append(Y[i:i+seq_len])
    if not X_seq:
        X_seq.append(X[:seq_len])
        Y_seq.append(Y[:seq_len])
    
    batches = []
    for i in range(0, len(X_seq), batch_size):
        x_b = torch.stack(X_seq[i:i+batch_size])
        y_b = torch.stack(Y_seq[i:i+batch_size])
        batches.append((x_b, y_b))
    return batches

def train():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    dataset = torch.load('synthesia_training_data/dataset_v5.pt', map_location='cpu')
    X_train, Y_train = dataset['train']['X'], dataset['train']['Y']
    X_val, Y_val = dataset['val']['X'], dataset['val']['Y']
    
    train_batches = make_batches(X_train, Y_train, batch_size=4, seq_len=32)
    val_batches = make_batches(X_val, Y_val, batch_size=4, seq_len=32)
    
    model = LearnedTemporalMotionModelV5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Resume test first
    chk = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': 0}
    torch.save(chk, 'resume_test.pt')
    chk_loaded = torch.load('resume_test.pt')
    model.load_state_dict(chk_loaded['model'])
    optimizer.load_state_dict(chk_loaded['opt'])
    print('Resume test passed')
    
    epochs = 250
    best_val_loss = float('inf')
    checkpoint_path = 'backend/training/checkpoints/motion_v5_best.pt'
    os.makedirs('backend/training/checkpoints', exist_ok=True)
    
    initial_loss = None
    final_loss = None
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_batches:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_batches)
        
        if epoch == 0: initial_loss = epoch_loss
        final_loss = epoch_loss
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_batches:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
        val_loss /= len(val_batches)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, checkpoint_path)
            
    t1 = time.time()
    print(f'Training complete in {t1 - t0:.2f}s')
    
    # Verification
    chk = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chk['model'])
    model.eval()
    
    file_size = os.path.getsize(checkpoint_path)
    sha256 = hashlib.sha256(open(checkpoint_path, 'rb').read()).hexdigest()
    num_params = sum(p.numel() for p in model.parameters())
    
    # Latency eval
    xb, yb = val_batches[0]
    xb = xb.to(device)
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            t_start = time.time()
            out = model(xb)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append((time.time() - t_start)*1000)
    
    latencies = np.array(latencies)
    
    print(f'Initial Loss: {initial_loss:.6f}')
    print(f'Final Train Loss: {final_loss:.6f}')
    print(f'Best Val Loss: {best_val_loss:.6f}')
    print(f'Params: {num_params}')
    print(f'Size: {file_size}')
    print(f'SHA256: {sha256}')
    print(f'Output Shape: {out.shape}')
    print(f'NaN: {torch.isnan(out).any().item()}')
    print(f'Inf: {torch.isinf(out).any().item()}')
    print(f'Latency Mean: {latencies.mean():.2f}ms')
    print(f'Latency p50: {np.percentile(latencies, 50):.2f}ms')
    print(f'Latency p95: {np.percentile(latencies, 95):.2f}ms')
    print(f'Latency Max: {latencies.max():.2f}ms')

if __name__ == '__main__':
    train()
