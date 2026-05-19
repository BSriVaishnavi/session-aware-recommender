"""
Step 3b: GRU model for trajectory prediction.
Given the first N tracks of a session, predict which trajectory
archetype the session will follow.

Run: python gru_trainer.py
Input:  data/sessions.csv, data/session_arcs_clustered.csv
Output: models/gru_model.pt, models/gru_config.pkl
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import mlflow
import mlflow.pytorch
import os

DATA_DIR   = "../data"
MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

MOOD_FEATURES  = ["valence", "energy", "danceability", "acousticness", "tempo"]
SEED_LENGTH    = 3       # use first 3 tracks to predict trajectory
EMBEDDING_DIM  = 64
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
DROPOUT        = 0.3
BATCH_SIZE     = 256
EPOCHS         = 30
LR             = 1e-3
N_CLUSTERS     = 8


class SessionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.dropout(h_n[-1])
        return self.classifier(out)


def build_sequences(sessions_df, arcs_df):
    """Build (seed_sequence, trajectory_label) pairs."""
    label_map = arcs_df.set_index("global_session_id")["trajectory_cluster"].to_dict()

    X, y = [], []
    for sid, grp in sessions_df.groupby("global_session_id"):
        if sid not in label_map:
            continue
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < SEED_LENGTH:
            continue
        seed = grp.iloc[:SEED_LENGTH][MOOD_FEATURES].values
        if np.isnan(seed).any():
            continue
        X.append(seed)
        y.append(label_map[sid])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    sessions = pd.read_csv(f"{DATA_DIR}/sessions.csv")
    arcs     = pd.read_csv(f"{DATA_DIR}/session_arcs_clustered.csv")
    print(f"Sessions: {len(sessions):,} tracks | Arcs: {len(arcs):,}")

    print("Building sequences...")
    X, y = build_sequences(sessions, arcs)
    print(f"Dataset: {len(X):,} samples | Classes: {np.unique(y)}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(SessionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SessionDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model     = TrajectoryGRU(len(MOOD_FEATURES), HIDDEN_DIM, NUM_LAYERS, N_CLUSTERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    mlflow.set_experiment("trajectory-gru")
    with mlflow.start_run():
        mlflow.log_params({
            "hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS,
            "dropout": DROPOUT, "lr": LR, "epochs": EPOCHS,
            "seed_length": SEED_LENGTH, "n_clusters": N_CLUSTERS,
        })

        best_val_acc = 0
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(va_loss)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss,   "val_acc": va_acc,
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save(model.state_dict(), f"{MODELS_DIR}/gru_model.pt")

        mlflow.log_metric("best_val_acc", best_val_acc)
        print(f"\nBest val accuracy: {best_val_acc:.4f}")

    config = {
        "input_dim": len(MOOD_FEATURES),
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "num_classes": N_CLUSTERS,
        "dropout": DROPOUT,
        "mood_features": MOOD_FEATURES,
        "seed_length": SEED_LENGTH,
    }
    joblib.dump(config, f"{MODELS_DIR}/gru_config.pkl")
    print(f"Saved GRU model → {MODELS_DIR}/gru_model.pt")


if __name__ == "__main__":
    main()
