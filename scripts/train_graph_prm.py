#!/usr/bin/env python3
"""
Train 18-dim Graph PRM.

Usage:
    python train_graph_prm.py \
        --data-dir /data/DRKG/KGSA/Stage5_Agent/prm_data/ \
        --output-dir /data/DRKG/KGSA/Stage5_Agent/models/
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from collections import Counter


class PreferenceDataset(Dataset):
    def __init__(self, X_better, X_worse):
        self.X_better = torch.FloatTensor(X_better)
        self.X_worse = torch.FloatTensor(X_worse)

    def __len__(self):
        return len(self.X_better)

    def __getitem__(self, idx):
        return self.X_better[idx], self.X_worse[idx]


class RewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def preference_loss(reward_better, reward_worse, margin=0.0):
    diff = reward_better - reward_worse - margin
    return -torch.log(torch.sigmoid(diff) + 1e-8).mean()


def extract_features(state: dict, feature_names: list, feature_norms: dict) -> np.ndarray:
    features = []
    for name in feature_names:
        value = state.get(name, 0)
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        norm = feature_norms.get(name, 1.0)
        features.append(float(value) / norm)
    return np.array(features, dtype=np.float32)


def prepare_dataset(preferences: list, feature_names: list, feature_norms: dict):
    X_better, X_worse, pref_types = [], [], []
    for pref in preferences:
        X_better.append(extract_features(pref.get("state_better", {}), feature_names, feature_norms))
        X_worse.append(extract_features(pref.get("state_worse", {}), feature_names, feature_norms))
        pref_types.append(pref.get("preference_type", "unknown"))
    return np.array(X_better), np.array(X_worse), pref_types


def train_epoch(model, loader, optimizer, device, margin=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, xw in loader:
        xb, xw = xb.to(device), xw.to(device)
        optimizer.zero_grad()
        rb, rw = model(xb), model(xw)
        loss = preference_loss(rb, rw, margin)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (rb > rw).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    diffs = []
    with torch.no_grad():
        for xb, xw in loader:
            xb, xw = xb.to(device), xw.to(device)
            rb, rw = model(xb), model(xw)
            correct += (rb > rw).sum().item()
            total += len(xb)
            diffs.extend((rb - rw).cpu().tolist())
    return correct / total, diffs


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3,
                weight_decay=1e-4, margin=0.0, patience=20, device="cpu"):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'Epoch':>6} {'Loss':>10} {'Train':>10} {'Val':>10} {'LR':>10}")
    print("-" * 50)

    for epoch in range(epochs):
        tloss, tacc = train_epoch(model, train_loader, optimizer, device, margin)
        vacc, _ = evaluate(model, val_loader, device)
        scheduler.step(vacc)

        history["train_loss"].append(tloss)
        history["train_acc"].append(tacc)
        history["val_acc"].append(vacc)

        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"{epoch+1:>6} {tloss:>10.4f} {tacc:>10.4f} {vacc:>10.4f} {lr_now:>10.2e}")

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    return model, best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description="Train 18-dim Graph PRM")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=str, default="128,64,32")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data + metadata
    with open(data_dir / "train_preferences.json") as f:
        train_data = json.load(f)
    with open(data_dir / "val_preferences.json") as f:
        val_data = json.load(f)

    train_prefs = train_data["preferences"]
    val_prefs = val_data["preferences"]
    meta = train_data.get("metadata", {})

    feature_names = meta.get("feature_names", [])
    feature_norms = meta.get("feature_norms", {})
    if not feature_names:
        raise ValueError("No feature_names in metadata")

    print(f"Feature dim: {len(feature_names)}")
    print(f"Train pairs: {len(train_prefs)}, Val pairs: {len(val_prefs)}")

    if not train_prefs or not val_prefs:
        raise ValueError(f"Insufficient data: train={len(train_prefs)}, val={len(val_prefs)}")

    train_types = Counter(p["preference_type"] for p in train_prefs)
    print("Train type distribution:")
    for pt, c in train_types.most_common(5):
        print(f"  {pt}: {c}")

    # Prepare datasets
    Xb_tr, Xw_tr, _ = prepare_dataset(train_prefs, feature_names, feature_norms)
    Xb_va, Xw_va, _ = prepare_dataset(val_prefs, feature_names, feature_norms)
    feature_dim = Xb_tr.shape[1]

    train_loader = DataLoader(PreferenceDataset(Xb_tr, Xw_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(PreferenceDataset(Xb_va, Xw_va), batch_size=args.batch_size)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    model = RewardModel(input_dim=feature_dim, hidden_dims=hidden_dims)
    print(f"Model: {feature_dim} -> {hidden_dims} -> 1  ({sum(p.numel() for p in model.parameters()):,} params)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, best_acc, history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        margin=args.margin, patience=args.patience, device=device,
    )

    # Save checkpoint
    model_path = output_dir / "graph_prm.pt"
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "feature_dim": feature_dim,
        "hidden_dims": hidden_dims,
        "feature_names": feature_names,
        "feature_norms": feature_norms,
        "best_val_acc": best_acc,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save report
    _, val_diffs = evaluate(model.to(device), val_loader, device)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "data": {"train_pairs": len(train_prefs), "val_pairs": len(val_prefs),
                 "feature_dim": feature_dim, "feature_names": feature_names},
        "results": {"best_val_accuracy": float(best_acc),
                     "reward_diff_mean": float(np.mean(val_diffs)),
                     "reward_diff_std": float(np.std(val_diffs))},
        "history": {k: [float(x) for x in v] for k, v in history.items()},
    }
    with open(output_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
