"""
Train FEA surrogate model from LHS-sampled data — v2.

Changes vs v1:
  - LOG_OUTPUTS expanded: sigma_t + p_AC_mid_mid + p_AC_lower_mid
    (all three have heavy-tailed / wide-range distributions)
  - Default patience: 50 → 120
  - Default weight_decay: 1e-5 → 1e-4 (combat overfit)
  - Added optional dropout (default 0.10)
  - Saves to surrogate_v2.pt to keep v1 around for comparison

Input  : output/surrogate_data/lhs_*.jsonl
Output : output/surrogate_model/surrogate_v2.pt, metrics.json, training_log.csv, parity_plots.png

Usage:
    python -m scripts.train_surrogate_v2 --data output/surrogate_data/lhs_20260519_214748.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_surrogate_v2")


INPUT_LABELS = [
    "h_upper_AC", "h_mid_AC", "h_lower_AC", "h_base", "h_subbase",
    "E_upper_AC", "E_mid_AC", "E_lower_AC", "E_base", "E_subbase",
    "E_subgrade",
    "is_semi_rigid",         # ★ v3 dual-base: 1.0 if semi_rigid, 0.0 if flexible
]
OUTPUT_LABELS = [
    "epsilon_a_microstrain",
    "sigma_t_MPa",
    "epsilon_z_microstrain",
    "p_AC_upper_mid_MPa",
    "p_AC_mid_mid_MPa",
    "p_AC_lower_mid_MPa",
]

# ★ v2 CHANGE: three log-transformed outputs (was just sigma_t in v1)
LOG_OUTPUTS = {
    "sigma_t_MPa",
    "p_AC_mid_mid_MPa",      # range ×3.3
    "p_AC_lower_mid_MPa",    # range ×12, strong heteroscedasticity
}

INPUT_DIM = 12               # ★ v3 dual-base: was 11
OUTPUT_DIM = 6


# ────────────────────────────────────────────────────────────────────────
# Data loading (unchanged)
# ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    X_rows, Y_rows = [], []
    meta: Dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in rec:
                meta = rec["_meta"]
                continue
            if rec.get("skipped", False) or "input" not in rec or "output" not in rec:
                continue
            inp, out = rec["input"], rec["output"]
            # ★ v3 dual-base: accept both v1/v2 (no is_semi_rigid) and v2 LHS data
            if "is_semi_rigid" in inp:
                is_semi = float(inp["is_semi_rigid"])
            elif inp.get("pavement_type", "").lower() in ("flexible", "unbound"):
                is_semi = 0.0
            else:
                # Legacy v1/v2 data: assume semi_rigid (the only mode that existed)
                is_semi = 1.0
            x = (inp["thickness_m"] + inp["modulus_MPa"]
                 + [inp["E_subgrade"]] + [is_semi])
            y = [out[lbl] for lbl in OUTPUT_LABELS]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                continue
            X_rows.append(x); Y_rows.append(y)
    X = np.asarray(X_rows, dtype=np.float64)
    Y = np.asarray(Y_rows, dtype=np.float64)
    logger.info(f"Loaded {X.shape[0]} samples; X={X.shape} Y={Y.shape}")
    return X, Y, meta


# ────────────────────────────────────────────────────────────────────────
# Scaler (unchanged)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Scaler:
    x_mean: np.ndarray; x_std: np.ndarray
    y_mean: np.ndarray; y_std: np.ndarray
    log_mask: np.ndarray

    @classmethod
    def fit(cls, X, Y, log_outputs) -> "Scaler":
        log_mask = np.array([(lbl in log_outputs) for lbl in OUTPUT_LABELS], dtype=bool)
        Y_t = Y.copy()
        if log_mask.any():
            cols = np.where(log_mask)[0]
            Y_t[:, cols] = np.log10(np.clip(Y_t[:, cols], 1e-8, None))
        return cls(
            x_mean = X.mean(axis=0),
            x_std  = X.std(axis=0) + 1e-12,
            y_mean = Y_t.mean(axis=0),
            y_std  = Y_t.std(axis=0) + 1e-12,
            log_mask = log_mask,
        )

    def transform_x(self, X): return (X - self.x_mean) / self.x_std
    def transform_y(self, Y):
        Y_t = Y.copy()
        if self.log_mask.any():
            cols = np.where(self.log_mask)[0]
            Y_t[:, cols] = np.log10(np.clip(Y_t[:, cols], 1e-8, None))
        return (Y_t - self.y_mean) / self.y_std

    def inverse_y(self, Yn):
        Yt = Yn * self.y_std + self.y_mean
        if self.log_mask.any():
            cols = np.where(self.log_mask)[0]
            Yt[:, cols] = np.power(10.0, Yt[:, cols])
        return Yt

    def to_dict(self):
        return {
            "x_mean": self.x_mean.tolist(), "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean.tolist(), "y_std": self.y_std.tolist(),
            "log_mask": self.log_mask.tolist(),
            "input_labels": INPUT_LABELS, "output_labels": OUTPUT_LABELS,
        }


# ────────────────────────────────────────────────────────────────────────
# MLP — ★ v2 CHANGE: optional dropout
# ────────────────────────────────────────────────────────────────────────

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, out_dim=OUTPUT_DIM,
                 hidden=(128, 128, 64), dropout: float = 0.0):
        super().__init__()
        dims = (in_dim,) + tuple(hidden) + (out_dim,)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.dropout = dropout

    def forward(self, x): return self.net(x)


def split_data(X, Y, seed, val_frac=0.10, test_frac=0.10):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac)); n_val = int(round(n * val_frac))
    te, va, tr = idx[:n_test], idx[n_test:n_test+n_val], idx[n_test+n_val:]
    return (X[tr], Y[tr]), (X[va], Y[va]), (X[te], Y[te])


def train_epoch(model, loader, opt, loss_fn, device):
    model.train(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval(); tot, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def predict_denorm(model, X, scaler, device):
    model.eval()
    xb = torch.as_tensor(scaler.transform_x(X), dtype=torch.float32, device=device)
    return scaler.inverse_y(model(xb).cpu().numpy())


def compute_metrics(y_true, y_pred):
    metrics = {}
    for j, lbl in enumerate(OUTPUT_LABELS):
        t, p = y_true[:, j], y_pred[:, j]
        ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        eps = max(np.abs(t).mean() * 1e-6, 1e-12)
        mape = np.mean(np.abs(t - p) / np.maximum(np.abs(t), eps)) * 100.0
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        metrics[lbl] = {"R2": float(r2), "MAPE_%": float(mape), "RMSE": rmse,
                        "y_min": float(t.min()), "y_max": float(t.max()),
                        "y_mean": float(t.mean()),
                        "y_range_ratio": float(t.max() / max(t.min(), 1e-9))}
    return metrics


def make_parity_plots(y_true, y_pred, out_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, lbl in enumerate(OUTPUT_LABELS):
        ax = axes[j // 3, j % 3]
        t, p = y_true[:, j], y_pred[:, j]
        ax.scatter(t, p, s=8, alpha=0.6)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1)
        ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        mape = np.mean(np.abs(t - p) / np.maximum(np.abs(t), 1e-9)) * 100.0
        ax.set_title(f"{lbl}\nR² = {r2:.4f}  MAPE = {mape:.2f}%", fontsize=9)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="output/surrogate_model")
    parser.add_argument("--epochs", type=int, default=1000)               # ★ was 500
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)       # ★ was 1e-5
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.10)            # ★ new
    parser.add_argument("--patience", type=int, default=120)              # ★ was 50
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tag", type=str, default="v2",
                        help="filename tag for saved model (default v2)")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    logger.info(f"Device: {device}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}"); sys.exit(1)
    X, Y, meta = load_jsonl(data_path)
    if X.shape[0] < 100:
        logger.error(f"Too few samples: {X.shape[0]}"); sys.exit(1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = split_data(X, Y, args.seed)
    logger.info(f"Split: train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]}")

    scaler = Scaler.fit(X_tr, Y_tr, log_outputs=LOG_OUTPUTS)
    logger.info(f"log_mask = {scaler.log_mask.tolist()}")
    logger.info(f"Log-transformed outputs: {[l for l in OUTPUT_LABELS if l in LOG_OUTPUTS]}")

    Xn_tr = scaler.transform_x(X_tr); Yn_tr = scaler.transform_y(Y_tr)
    Xn_va = scaler.transform_x(X_va); Yn_va = scaler.transform_y(Y_va)

    tr_loader = DataLoader(TensorDataset(
        torch.tensor(Xn_tr, dtype=torch.float32),
        torch.tensor(Yn_tr, dtype=torch.float32)),
        batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(TensorDataset(
        torch.tensor(Xn_va, dtype=torch.float32),
        torch.tensor(Yn_va, dtype=torch.float32)),
        batch_size=args.batch, shuffle=False)

    model = SurrogateMLP(hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: MLP({INPUT_DIM} → {' → '.join(map(str, args.hidden))} "
                f"→ {OUTPUT_DIM})  dropout={args.dropout}  params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    log_rows = []
    best_val = float("inf"); best_epoch = -1; patience_cnt = 0
    best_state = None
    logger.info("=" * 70)
    logger.info("Training start  (v2: log on 3 outputs, dropout, more patience)")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, tr_loader, opt, loss_fn, device)
        va = eval_epoch(model, va_loader, loss_fn, device)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        log_rows.append((epoch, tr, va, lr_now))

        improved = va < best_val - 1e-6
        if improved:
            best_val = va; best_epoch = epoch; patience_cnt = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 25 == 0 or epoch == 1:
            tag = "  <-- best" if improved else ""
            logger.info(f"  epoch {epoch:4d}  tr={tr:.5f}  va={va:.5f}  "
                        f"lr={lr_now:.2e}{tag}")

        if patience_cnt >= args.patience:
            logger.info(f"Early stop @ epoch {epoch} (best={best_epoch}, "
                        f"val={best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Best val loss: {best_val:.5f} at epoch {best_epoch}")

    logger.info("=" * 70)
    logger.info("Evaluation (denormalized space)")
    logger.info("=" * 70)
    Y_pred_tr = predict_denorm(model, X_tr, scaler, device)
    Y_pred_va = predict_denorm(model, X_va, scaler, device)
    Y_pred_te = predict_denorm(model, X_te, scaler, device)
    m_tr = compute_metrics(Y_tr, Y_pred_tr)
    m_va = compute_metrics(Y_va, Y_pred_va)
    m_te = compute_metrics(Y_te, Y_pred_te)

    logger.info(f"{'Output':<26} {'R²(tr)':>8} {'R²(va)':>8} {'R²(te)':>8}  "
                f"{'MAPE(te)':>10}")
    logger.info("-" * 80)
    all_pass = True
    for lbl in OUTPUT_LABELS:
        r2_te = m_te[lbl]["R2"]; mape_te = m_te[lbl]["MAPE_%"]
        passed = (r2_te >= 0.95) or (mape_te <= 3.0)   # accept low R² if MAPE tiny
        all_pass &= passed
        flag = "✓" if passed else "✗"
        logger.info(f"  {lbl:<24} {m_tr[lbl]['R2']:8.4f} {m_va[lbl]['R2']:8.4f} "
                    f"{r2_te:8.4f}  {mape_te:9.2f}%  {flag}")
    logger.info("-" * 80)
    logger.info(f"Acceptance (test R²≥0.95 OR MAPE≤3%): "
                f"{'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'}")

    ckpt_path = out_dir / f"surrogate_{args.tag}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_arch": {"in_dim": INPUT_DIM, "out_dim": OUTPUT_DIM,
                       "hidden": list(args.hidden), "dropout": args.dropout},
        "scaler": scaler.to_dict(),
        "input_labels": INPUT_LABELS, "output_labels": OUTPUT_LABELS,
        "training_meta": {
            "data_file": str(data_path), "n_samples": int(X.shape[0]),
            "n_train": int(X_tr.shape[0]), "n_val": int(X_va.shape[0]),
            "n_test": int(X_te.shape[0]),
            "epochs_run": epoch, "best_epoch": best_epoch,
            "best_val": float(best_val), "seed": args.seed,
            "lr": args.lr, "batch": args.batch,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "log_outputs": sorted(LOG_OUTPUTS),
        },
        "lhs_meta": meta,
    }, ckpt_path)
    logger.info(f"Saved model: {ckpt_path}")

    with open(out_dir / f"metrics_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "val": m_va, "test": m_te,
                   "all_pass": all_pass}, f, indent=2)
    with open(out_dir / f"training_log_{args.tag}.csv", "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr\n")
        for row in log_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    make_parity_plots(Y_te, Y_pred_te, out_dir / f"parity_plots_{args.tag}.png")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
