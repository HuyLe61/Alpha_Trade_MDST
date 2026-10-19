"""
Export a trained PPO checkpoint to a small, torch-free .npz file.

The live worker runs on a cron (GitHub Actions) and only needs the actor's
deterministic mean action: tanh(W2 @ relu(W1 @ obs + b1) + b2). Shipping the
weights as numpy arrays keeps the worker install to a few seconds instead of
pulling ~800MB of PyTorch on every run.

If the checkpoint is missing the z-score stats (feat_mean / feat_std), they are
rebuilt exactly the way AlphaTradeEnv computes them: concatenate every training
ticker's feature rows (2009-2019 split) and take the per-column mean / std.

Run (needs torch, so use the training environment):
    python -m trader.export_model --checkpoint models/ppo_trading.pt
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import process_data_with_indicators, split_data  # noqa: E402
from trader.config import TICKERS  # noqa: E402

DEFAULT_OUT = Path(__file__).parent / "model" / "policy.npz"


def rebuild_feature_stats(data_dir: Path):
    """Mirror train.ipynb + AlphaTradeEnv.__init__ to recover feat_mean / feat_std."""
    stock_data = {}
    for ticker in TICKERS:
        df = pd.read_csv(data_dir / f"{ticker}.csv", skiprows=2, index_col="Date", parse_dates=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        stock_data[ticker] = df
    stock_data = process_data_with_indicators(stock_data)
    training_data, _, _ = split_data(stock_data)
    training_data = {t: df for t, df in training_data.items() if not df.empty}

    all_feat = np.concatenate([df.values for df in training_data.values()], axis=0)
    feat_mean = np.mean(all_feat, axis=0).astype(np.float32)
    feat_std = np.std(all_feat, axis=0).astype(np.float32)
    feat_std[feat_std < 1e-8] = 1.0
    return feat_mean, feat_std


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ppo_trading.pt"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["policy_state_dict"]

    feat_mean, feat_std = ckpt.get("feat_mean"), ckpt.get("feat_std")
    if feat_mean is None or feat_std is None:
        print("Checkpoint has no feat_mean/feat_std; rebuilding from training CSVs")
        feat_mean, feat_std = rebuild_feature_stats(Path(args.data_dir))

    arrays = {
        "w1": sd["actor.0.weight"].numpy().astype(np.float32),
        "b1": sd["actor.0.bias"].numpy().astype(np.float32),
        "w2": sd["actor.2.weight"].numpy().astype(np.float32),
        "b2": sd["actor.2.bias"].numpy().astype(np.float32),
        "feat_mean": np.asarray(feat_mean, dtype=np.float32),
        "feat_std": np.asarray(feat_std, dtype=np.float32),
        "obs_dim": np.int64(ckpt["obs_shape"]),
        "act_dim": np.int64(ckpt["action_shape"]),
    }
    digest = hashlib.sha256(b"".join(a.tobytes() for a in arrays.values())).hexdigest()[:12]
    arrays["version"] = np.array(f"{Path(args.checkpoint).stem}-{digest}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"Wrote {out} (version {arrays['version']}, obs={arrays['obs_dim']}, act={arrays['act_dim']})")

    # Sanity check: numpy forward pass must match torch on random observations.
    from agent.actor_critic import ActorCritic
    from trader.policy import Policy

    net = ActorCritic(int(ckpt["obs_shape"]), int(ckpt["action_shape"]))
    net.load_state_dict(sd)
    net.eval()
    obs = np.random.default_rng(0).uniform(-3, 3, size=(8, int(ckpt["obs_shape"]))).astype(np.float32)
    with torch.no_grad():
        dist, _ = net(torch.from_numpy(obs))
        expected = torch.tanh(dist.mean).numpy()
    policy = Policy(out)
    got = np.stack([policy.act(o) for o in obs])
    err = float(np.max(np.abs(expected - got)))
    print(f"numpy vs torch max abs diff: {err:.2e}")
    if err > 1e-4:
        raise SystemExit("Exported policy does not match torch output")


if __name__ == "__main__":
    main()
