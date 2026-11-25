"""ETTh1 loading, chronological splitting, scaling, and sliding windows."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"

def make_loaders(csv_path, batch_size=64, seq_len=96, horizon=24):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    values = df[FEATURES].to_numpy(dtype=np.float32)
    n = len(values)
    train_end, val_end = int(n * .70), int(n * .80)
    scaler = StandardScaler().fit(values[:train_end])
    scaled = scaler.transform(values).astype(np.float32)
    target_idx = FEATURES.index(TARGET)

    def windows(start, stop):
        xs, ys = [], []
        # Include history from the immediately preceding partition boundary.
        for end in range(max(seq_len, start), stop - horizon + 1):
            xs.append(scaled[end-seq_len:end])
            ys.append(scaled[end:end+horizon, target_idx])
        return torch.tensor(np.asarray(xs)), torch.tensor(np.asarray(ys))

    train = windows(seq_len, train_end)
    val = windows(train_end, val_end)
    test = windows(val_end, n)
    def loader(pair, shuffle):
        return DataLoader(TensorDataset(*pair), batch_size=batch_size, shuffle=shuffle)
    return loader(train, True), loader(val, False), loader(test, False), scaler, target_idx
