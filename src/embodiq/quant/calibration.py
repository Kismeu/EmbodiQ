"""Bounded, deterministic activation collection on training observations."""
from __future__ import annotations
import hashlib
import torch


class ActivationCollector:
    def __init__(self, modules, *, max_rows=64, rows_per_call=8, seed=42):
        if max_rows < 1 or rows_per_call < 1:
            raise ValueError("Sample budgets must be positive")
        self.modules = modules
        self.max_rows, self.rows_per_call, self.seed = max_rows, rows_per_call, seed
        self.records = {}
        self.handles = []

    def __enter__(self):
        if self.handles: raise RuntimeError("Collector already active")
        for name, module in self.modules.items():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
            self.records[name] = dict(calls=0, rows_seen=0, candidates_seen=0, abs_sum=None,
                                      samples=[], generator=generator)
            self.handles.append(module.register_forward_pre_hook(self._hook(name)))
        return self

    def _hook(self, name):
        def hook(module, args):
            x = args[0].detach().reshape(-1, module.in_features)
            if x.numel() == 0: return
            if not torch.isfinite(x).all(): raise ValueError(f"Non-finite calibration input: {name}")
            row = self.records[name]
            total = x.float().abs().sum(0).cpu()
            row["abs_sum"] = total if row["abs_sum"] is None else row["abs_sum"] + total
            row["calls"] += 1; row["rows_seen"] += len(x)
            indices = torch.linspace(0, len(x) - 1, min(len(x), self.rows_per_call), device=x.device).long()
            for sample in x.index_select(0, indices).float().cpu():
                row["candidates_seen"] += 1
                if len(row["samples"]) < self.max_rows:
                    row["samples"].append(sample.clone())
                else:
                    idx = int(torch.randint(row["candidates_seen"], (1,), generator=row["generator"]))
                    if idx < self.max_rows: row["samples"][idx] = sample.clone()
        return hook

    def __exit__(self, *_):
        for handle in self.handles: handle.remove()
        self.handles.clear()

    def export(self, *, allow_missing: bool = False):
        missing = [name for name, row in self.records.items() if row["calls"] == 0]
        if missing and not allow_missing:
            raise ValueError(f"No calibration activations for: {missing}")
        return {name: dict(mean_abs=row["abs_sum"] / row["rows_seen"],
                           samples=torch.stack(row["samples"]), calls=row["calls"],
                           rows_seen=row["rows_seen"], candidates_seen=row["candidates_seen"])
                for name, row in self.records.items() if row["calls"] > 0}
