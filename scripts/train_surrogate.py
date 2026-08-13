"""
Train FEA surrogate model from LHS-sampled data.

Input  : output/surrogate_data/lhs_*.jsonl
Output : output/surrogate_model/
           surrogate_v1.pt        (model weights + arch + scaler params, single file)
           metrics.json           (R², MAPE per output, train/val/test)
           training_log.csv       (loss curves)
           parity_plots.png       (6 subplots, predicted vs true)

Usage:
    python -m scripts.train_surrogate --data output/surrogate_data/lhs_20260519_214748.jsonl
    python -m scripts.train_surrogate --data ...  --epochs 500 --batch 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
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
logger = logging.getLogger("train_surrogate")


# ────────────────────────────────────────────────────────────────────────
# Constants — must match LHS sampler and env.py
# ────────────────────────────────────────────────────────────────────────

INPUT_LABELS = [
    "h_upper_AC", "h_mid_AC", "h_lower_AC", "h_base", "h_subbase",
    "E_upper_AC", "E_mid_AC", "E_lower_AC", "E_base", "E_subbase",
    "E_subgrade",
]
OUTPUT_LABELS = [
    "epsilon_a_microstrain",
    "sigma_t_MPa",
    "epsilon_z_microstrain",
    "p_AC_upper_mid_MPa",
    "p_AC_mid_mid_MPa",
    "p_AC_lower_mid_MPa",
]

# Outputs whose distribution is heavy-tailed — apply log10 before z-score
LOG_OUTPUTS = {"sigma_t_MPa"}

INPUT_DIM = 11
OUTPUT_DIM = 6


# ────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Read LHS JSONL → (X [N,11], Y [N,6], meta dict)."""
    X_rows: List[List[float]] = []
    Y_rows: List[List[float]] = []
    meta: Dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"  line {line_no}: JSON parse error, skipping: {e}")
                continue
            if "_meta" in rec:
                meta = rec["_meta"]
                continue
            if rec.get("skipped", False):
                continue
            if "input" not in rec or "output" not in rec:
                continue

            inp = rec["input"]
            out = rec["output"]
            x = inp["thickness_m"] + inp["modulus_MPa"] + [inp["E_subgrade"]]
            y = [out[lbl] for lbl in OUTPUT_LABELS]
            # Skip anything containing NaN/inf
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                continue
            X_rows.append(x)
            Y_rows.append(y)

    X = np.asarray(X_rows, dtype=np.float64)
    Y = np.asarray(Y_rows, dtype=np.float64)
    logger.info(f"Loaded {X.shape[0]} samples from {path.name}")
    logger.info(f"  X shape={X.shape}  Y shape={Y.shape}")
    return X, Y, meta


# ────────────────────────────────────────────────────────────────────────
# Scaler (with log transform for selected outputs)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class Scaler:
    """Stores per-feature normalization parameters. Pickle-safe (dataclass)."""
    x_mean: np.ndarray
    x_std:  np.ndarray
    y_mean: np.ndarray
    y_std:  np.ndarray
    log_mask: np.ndarray   # bool [OUTPUT_DIM]

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, log_outputs: set) -> "Scaler":
        log_mask = np.array(
            [(lbl in log_outputs) for lbl in OUTPUT_LABELS],
            dtype=bool,
        )
        # Apply log10 to log_mask columns BEFORE computing mean/std
        Y_t = Y.copy()
        if log_mask.any():
            cols = np.where(log_mask)[0]
            # Guard against zeros — clip to a small positive value
            Y_t[:, cols] = np.log10(np.clip(Y_t[:, cols], 1e-8, None))
        return cls(
            x_mean = X.mean(axis=0),
            x_std  = X.std(axis=0) + 1e-12,
            y_mean = Y_t.mean(axis=0),
            y_std  = Y_t.std(axis=0) + 1e-12,
            log_mask = log_mask,
        )

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def transform_y(self, Y: np.ndarray) -> np.ndarray:
        Y_t = Y.copy()
        if self.log_mask.any():
            cols = np.where(self.log_mask)[0]
            Y_t[:, cols] = np.log10(np.clip(Y_t[:, cols], 1e-8, None))
        return (Y_t - self.y_mean) / self.y_std

    def inverse_y(self, Y_norm: np.ndarray) -> np.ndarray:
        Y_t = Y_norm * self.y_std + self.y_mean
        if self.log_mask.any():
            cols = np.where(self.log_mask)[0]
            Y_t[:, cols] = np.power(10.0, Y_t[:, cols])
        return Y_t

    def to_dict(self) -> Dict:
        return {
            "x_mean":   self.x_mean.tolist(),
            "x_std":    self.x_std.tolist(),
            "y_mean":   self.y_mean.tolist(),
            "y_std":    self.y_std.tolist(),
            "log_mask": self.log_mask.tolist(),
            "input_labels":  INPUT_LABELS,
            "output_labels": OUTPUT_LABELS,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Scaler":
        return cls(
            x_mean   = np.asarray(d["x_mean"],   dtype=np.float64),
            x_std    = np.asarray(d["x_std"],    dtype=np.float64),
            y_mean   = np.asarray(d["y_mean"],   dtype=np.float64),
            y_std    = np.asarray(d["y_std"],    dtype=np.float64),
            log_mask = np.asarray(d["log_mask"], dtype=bool),
        )


# ────────────────────────────────────────────────────────────────────────
# MLP model
# ────────────────────────────────────────────────────────────────────────

class SurrogateMLP(nn.Module):
    """Plain MLP regressor. Operates in normalized space."""

    def __init__(self, in_dim: int = INPUT_DIM, out_dim: int = OUTPUT_DIM,
                 hidden: Tuple[int, ...] = (128, 128, 64)):
        super().__init__()
        dims = (in_dim,) + tuple(hidden) + (out_dim,)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)
        self.hidden_dims = tuple(hidden)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────

def split_data(X: np.ndarray, Y: np.ndarray, seed: int,
               val_frac: float = 0.10, test_frac: float = 0.10
               ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Random 80/10/10 split using numpy with fixed seed."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return ((X[train_idx], Y[train_idx]),
            (X[val_idx],   Y[val_idx]),
            (X[test_idx],  Y[test_idx]))


def train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        tot += loss.item() * xb.size(0); n += xb.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def predict_denorm(model, X: np.ndarray, scaler: Scaler, device) -> np.ndarray:
    model.eval()
    xb = torch.as_tensor(scaler.transform_x(X), dtype=torch.float32, device=device)
    yb = model(xb).cpu().numpy()
    return scaler.inverse_y(yb)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Per-output R², MAPE, RMSE."""
    metrics = {}
    for j, lbl in enumerate(OUTPUT_LABELS):
        t = y_true[:, j]
        p = y_pred[:, j]
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        # MAPE — guard against tiny y_true
        eps = max(np.abs(t).mean() * 1e-6, 1e-12)
        mape = np.mean(np.abs(t - p) / np.maximum(np.abs(t), eps)) * 100.0
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        metrics[lbl] = {"R2": float(r2), "MAPE_%": float(mape), "RMSE": rmse,
                        "y_min": float(t.min()), "y_max": float(t.max()),
                        "y_mean": float(t.mean())}
    return metrics


# ────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ────────────────────────────────────────────────────────────────────────

def make_parity_plots(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping parity plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for j, lbl in enumerate(OUTPUT_LABELS):
        ax = axes[j // 3, j % 3]
        t = y_true[:, j]; p = y_pred[:, j]
        ax.scatter(t, p, s=8, alpha=0.6)
        lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", lw=1)
        ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        ax.set_title(f"{lbl}\nR² = {r2:.4f}", fontsize=10)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved parity plots: {out_path}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train FEA surrogate model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to LHS jsonl file")
    parser.add_argument("--out-dir", type=str, default="output/surrogate_model")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 64])
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stop patience (epochs without val improvement)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # ── Device ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Seeds ──
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── Load data ──
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    X, Y, meta = load_jsonl(data_path)
    if X.shape[0] < 100:
        logger.error(f"Too few samples ({X.shape[0]}); need at least 100")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Split ──
    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = split_data(X, Y, args.seed)
    logger.info(f"Split: train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]}")

    # ── Fit scaler ON TRAINING SET ONLY ──
    scaler = Scaler.fit(X_tr, Y_tr, log_outputs=LOG_OUTPUTS)
    logger.info(f"Scaler fitted. log_mask = {scaler.log_mask.tolist()}")

    # Convert to normalized tensors
    Xn_tr = scaler.transform_x(X_tr); Yn_tr = scaler.transform_y(Y_tr)
    Xn_va = scaler.transform_x(X_va); Yn_va = scaler.transform_y(Y_va)
    Xn_te = scaler.transform_x(X_te); Yn_te = scaler.transform_y(Y_te)

    tr_ds = TensorDataset(torch.tensor(Xn_tr, dtype=torch.float32),
                          torch.tensor(Yn_tr, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(Xn_va, dtype=torch.float32),
                          torch.tensor(Yn_va, dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    # ── Model ──
    model = SurrogateMLP(hidden=tuple(args.hidden)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: MLP({INPUT_DIM} → {' → '.join(map(str, args.hidden))} → {OUTPUT_DIM})  "
                f"params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    # ── Train loop ──
    log_rows = []
    best_val = float("inf"); best_epoch = -1; patience_cnt = 0
    best_state = None
    logger.info("=" * 70)
    logger.info("Training start")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, tr_loader, opt, loss_fn, device)
        va_loss = eval_one_epoch (model, va_loader,      loss_fn, device)
        sched.step()
        lr_now = opt.param_groups[0]["lr"]
        log_rows.append((epoch, tr_loss, va_loss, lr_now))

        improved = va_loss < best_val - 1e-6
        if improved:
            best_val = va_loss; best_epoch = epoch; patience_cnt = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if epoch % 20 == 0 or epoch == 1:
            tag = "  <-- best" if improved else ""
            logger.info(f"  epoch {epoch:4d}  tr_loss={tr_loss:.5f}  "
                        f"va_loss={va_loss:.5f}  lr={lr_now:.2e}{tag}")

        if patience_cnt >= args.patience:
            logger.info(f"Early stop at epoch {epoch} (best epoch {best_epoch}, "
                        f"val={best_val:.5f}, no improvement for {patience_cnt} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Best val loss: {best_val:.5f} at epoch {best_epoch}")

    # ── Evaluate ──
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
    logger.info("-" * 70)
    all_pass = True
    for lbl in OUTPUT_LABELS:
        r2_te = m_te[lbl]["R2"]; mape_te = m_te[lbl]["MAPE_%"]
        passed = r2_te >= 0.95
        all_pass &= passed
        flag = "✓" if passed else "✗"
        logger.info(f"  {lbl:<24} {m_tr[lbl]['R2']:8.4f} {m_va[lbl]['R2']:8.4f} "
                    f"{r2_te:8.4f}  {mape_te:9.2f}%  {flag}")
    logger.info("-" * 70)
    logger.info(f"Acceptance (test R² ≥ 0.95 per output): "
                f"{'ALL PASS ✓' if all_pass else 'SOME FAIL ✗'}")

    # ── Save artefacts ──
    ckpt_path = out_dir / "surrogate_v1.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "model_arch": {
            "in_dim":  INPUT_DIM,
            "out_dim": OUTPUT_DIM,
            "hidden":  list(args.hidden),
        },
        "scaler":         scaler.to_dict(),
        "input_labels":   INPUT_LABELS,
        "output_labels":  OUTPUT_LABELS,
        "training_meta": {
            "data_file":   str(data_path),
            "n_samples":   int(X.shape[0]),
            "n_train":     int(X_tr.shape[0]),
            "n_val":       int(X_va.shape[0]),
            "n_test":      int(X_te.shape[0]),
            "epochs_run":  epoch,
            "best_epoch":  best_epoch,
            "best_val":    float(best_val),
            "seed":        args.seed,
            "lr":          args.lr,
            "batch":       args.batch,
            "weight_decay":args.weight_decay,
        },
        "lhs_meta": meta,
    }, ckpt_path)
    logger.info(f"Saved model: {ckpt_path}")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train": m_tr, "val": m_va, "test": m_te,
                   "all_pass_test_R2_0.95": all_pass}, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr\n")
        for row in log_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    logger.info(f"Saved training log: {log_path}")

    parity_path = out_dir / "parity_plots.png"
    make_parity_plots(Y_te, Y_pred_te, parity_path)

    logger.info("=" * 70)
    logger.info("DONE.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
