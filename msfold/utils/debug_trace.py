"""Shared debug tracing for gold-vs-release comparison.

Usage:
    from x.debug_trace import debug_trace
    debug_trace.enable("/path/to/debug_dir")
    debug_trace.log("key", tensor_or_value)
    ...
    debug_trace.save_checkpoint(step)  # save incremental checkpoint
    debug_trace.save()                 # final save
"""

import os
import pickle
import torch


class DebugTrace:
    def __init__(self):
        self._enabled = False
        self._data = {}
        self._dir = None
        self._step = -1

    def enable(self, dir_path: str):
        self._enabled = True
        self._dir = dir_path
        self._data = {}
        self._step = -1
        os.makedirs(dir_path, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, s: int):
        self._step = s

    def log(self, key: str, value):
        if not self._enabled:
            return
        if isinstance(value, torch.Tensor):
            self._data[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            self._data[key] = [
                v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for v in value
            ]
        else:
            self._data[key] = value

    def save_checkpoint(self, step: int):
        """Save incremental checkpoint for step comparison."""
        if not self._enabled:
            return
        path = os.path.join(self._dir, f"trace_step_{step:04d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._data, f)

    def save(self):
        if not self._enabled:
            return
        path = os.path.join(self._dir, "trace_final.pkl")
        with open(path, "wb") as f:
            pickle.dump(self._data, f)

    def disable(self):
        self._enabled = False
        self._data = {}
        self._dir = None
        self._step = -1


debug_trace = DebugTrace()
