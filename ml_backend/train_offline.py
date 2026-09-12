import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os
import pickle
from sklearn.metrics import precision_score, recall_score, confusion_matrix
torch.manual_seed(42)
np.random.seed(42)
# 10-Feature Contract[cite: 2]
FEATURES = [
    'sttl', 'sbytes', 'dbytes', 'sload', 'dur', 
    'rate', 'tcprtt', 'synack', 'ackdat', 'ct_dst_src_ltm'
]

# MLP Architecture[cite: 2]
class NIDS_MLP(nn.Module):
    def __init__(self):
        super(NIDS_MLP, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1) # Single output neuron, raw logit[cite: 2]

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    os.makedirs('artifacts', exist_ok=True)

    # Step 1: Data isolation[cite: 2]
    print("Loading training dataset...")
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    
    # Drop everything except contract features and label[cite: 2]
    X_train = train_df[FEATURES].copy()
    y_train = train_df['label'].astype('float32').values # float32 for BCEWithLogitsLoss[cite: 2]

    # Step 2: Scaling[cite: 2]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 4: Persist shared artifacts[cite: 2]
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    
    scaler_ranges = {
        feat: {"min": float(scaler.data_min_[i]), "max": float(scaler.data_max_[i])}
        for i, feat in enumerate(FEATURES)
    }
    with open('artifacts/scaler_ranges.json', 'w') as f:
        json.dump(scaler_ranges, f, indent=4)
    print("Saved scaler.pkl and scaler_ranges.json")

    # Step 5: Train[cite: 2]
    model = NIDS_MLP()
    criterion = nn.BCEWithLogitsLoss() # Operates on raw logit[cite: 2]
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam, lr=1e-3[cite: 2]
    
    dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print("Training base model...")
    model.train()
    for epoch in range(20): # 15-20 epochs[cite: 2]
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            
        print(f"Epoch {epoch+1}/20 - Loss: {epoch_loss/len(loader):.4f} - Acc: {correct/total:.4f}")

    torch.save(model.state_dict(), 'artifacts/base_model.pth')
    print("Saved base_model.pth")

    # Save initial memory buffer (250 benign, 250 attack)[cite: 2]
    benign_idx = np.where(y_train == 0.0)[0]
    attack_idx = np.where(y_train == 1.0)[0]
    
    init_benign = X_train_scaled[np.random.choice(benign_idx, 250, replace=False)]
    init_attack = X_train_scaled[np.random.choice(attack_idx, 250, replace=False)]
    
    buffer_data = {
        'benign': [(list(row), 0.0) for row in init_benign],
        'attack': [(list(row), 1.0) for row in init_attack]
    }
    with open('artifacts/initial_buffer.pkl', 'wb') as f:
        pickle.dump(buffer_data, f)

    # Evaluation on test set[cite: 2]
    print("\nEvaluating on test dataset...")
    test_df = pd.read_csv('UNSW_NB15_testing-set.csv')
    X_test = test_df[FEATURES].copy()
    y_test = test_df['label'].astype('float32').values
    
    X_test_scaled = scaler.transform(X_test)
    
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.FloatTensor(X_test_scaled))
        test_probs = torch.sigmoid(test_logits).numpy().squeeze()
        test_preds = (test_probs > 0.5).astype(float)
        
        # Standard 0.5 Boundary Metrics
        print(f"Test Accuracy (0.5 cutoff): {(test_preds == y_test).mean():.4f}")
        print(f"Precision (0.5 cutoff): {precision_score(y_test, test_preds):.4f}")
        print(f"Recall (0.5 cutoff): {recall_score(y_test, test_preds):.4f}")
        
        # Tri-Tier Specific Failure Metrics
        print("\n--- Production Architecture Metrics ---")
        attack_mask = (y_test == 1.0)
        benign_mask = (y_test == 0.0)
        
        P_attacks = test_probs[attack_mask]
        P_benign = test_probs[benign_mask]
        
        silently_missed = (P_attacks < 0.10).sum()
        falsely_blocked = (P_benign > 0.90).sum()
        
        print(f"Catastrophic Silent Misses (Attacks P < 0.10): {silently_missed} / {attack_mask.sum()} ({silently_missed/attack_mask.sum():.2%})")
        print(f"Unnecessary Auto-Blocks (Benign P > 0.90): {falsely_blocked} / {benign_mask.sum()} ({falsely_blocked/benign_mask.sum():.2%})")
        
        ambiguous_mask = (test_probs >= 0.10) & (test_probs <= 0.90)
        print(f"Traffic Escaping to Tier 3 SOC (0.10 <= P <= 0.90): {ambiguous_mask.mean() * 100:.2f}%")