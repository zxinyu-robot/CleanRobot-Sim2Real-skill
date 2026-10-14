"""A tiny CPU-only MLP for residual policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class MLP:
    def __init__(self, in_size: int, hidden: int, out_size: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.25
        self.w1 = rng.normal(0.0, scale / np.sqrt(in_size), (in_size, hidden))
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, scale / np.sqrt(hidden), (hidden, out_size))
        self.b2 = np.zeros(out_size, dtype=np.float64)

    def forward(self, x: FloatArray) -> FloatArray:
        single = x.ndim == 1
        batch = np.atleast_2d(x)
        hidden = np.tanh(batch @ self.w1 + self.b1)
        output = hidden @ self.w2 + self.b2
        return output[0] if single else output

    def fit(
        self,
        features: FloatArray,
        targets: FloatArray,
        steps: int = 400,
        batch_size: int = 64,
        lr: float = 0.05,
        seed: int = 0,
    ) -> list[float]:
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        n = len(features)
        batch_size = min(batch_size, n)
        for step in range(steps):
            index = rng.choice(n, size=batch_size, replace=False)
            x = features[index]
            y = targets[index]
            hidden_pre = x @ self.w1 + self.b1
            hidden = np.tanh(hidden_pre)
            prediction = hidden @ self.w2 + self.b2
            error = prediction - y
            loss = float(np.mean(error**2))
            grad_output = (2.0 / batch_size) * error
            grad_w2 = hidden.T @ grad_output
            grad_b2 = grad_output.sum(axis=0)
            grad_hidden = (grad_output @ self.w2.T) * (1.0 - hidden**2)
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1
            if step % 20 == 0:
                losses.append(loss)
                lr *= 0.98
        return losses

    def save(self, path: Path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    @classmethod
    def load(cls, path: Path) -> MLP:
        payload = np.load(path)
        model = cls(payload["w1"].shape[0], payload["w1"].shape[1], payload["w2"].shape[1])
        model.w1 = payload["w1"]
        model.b1 = payload["b1"]
        model.w2 = payload["w2"]
        model.b2 = payload["b2"]
        return model
