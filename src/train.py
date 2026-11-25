"""Train one ETTh1 forecasting model and report test metrics."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import make_loaders
from models import BaselineLSTM, BiLSTMAttention, TransformerForecaster

MODELS = {"lstm": BaselineLSTM, "attention": BiLSTMAttention, "transformer": TransformerForecaster}

def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
    return total / max(1, len(loader))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ETTh1.csv")
    p.add_argument("--model", choices=MODELS, default="lstm")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", default="artifacts")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, scaler, target_idx = make_loaders(args.data, args.batch_size)
    model = MODELS[args.model]().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_weights, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        tr = run_epoch(model, train, loss_fn, device, optimizer)
        vl = run_epoch(model, val, loss_fn, device)
        print(f"Epoch {epoch+1:02d}: train={tr:.6f} validation={vl:.6f}")
        if vl < best:
            best, best_weights, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break
    model.load_state_dict(best_weights)
    model.eval()
    preds, actual = [], []
    with torch.no_grad():
        for x, y in test:
            preds.append(model(x.to(device)).cpu().numpy())
            actual.append(y.numpy())
    pred = np.concatenate(preds).reshape(-1)
    truth = np.concatenate(actual).reshape(-1)
    mean, scale = scaler.mean_[target_idx], scaler.scale_[target_idx]
    pred, truth = pred * scale + mean, truth * scale + mean
    mse = mean_squared_error(truth, pred)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mean_absolute_error(truth, pred):.4f}")
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.model}_best.pth")
    np.savez(out / f"{args.model}_predictions.npz", actual=truth, predicted=pred)

if __name__ == "__main__":
    main()
