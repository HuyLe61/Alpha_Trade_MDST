"""Torch-free inference for the exported PPO actor (see export_model.py)."""

from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).parent / "model" / "policy.npz"


class Policy:
    def __init__(self, path=DEFAULT_PATH):
        z = np.load(path)
        self.w1, self.b1 = z["w1"], z["b1"]
        self.w2, self.b2 = z["w2"], z["b2"]
        self.feat_mean, self.feat_std = z["feat_mean"], z["feat_std"]
        self.obs_dim, self.act_dim = int(z["obs_dim"]), int(z["act_dim"])
        self.version = str(z["version"])
        self.bar_minutes = int(z["bar_minutes"]) if "bar_minutes" in z else 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action, same as AlpacaPPOTrader.predict: tanh(actor mean)."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Observation has {obs.shape[-1]} dims, policy expects {self.obs_dim}")
        h = np.maximum(self.w1 @ obs + self.b1, 0.0)
        return np.tanh(self.w2 @ h + self.b2).astype(np.float32)
