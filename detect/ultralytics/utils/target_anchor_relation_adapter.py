# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Extract a checkpoint state dict from common training payload layouts."""
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "adapter", "adapter_state_dict", "model_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported adapter payload type: {type(payload)!r}")


def _strip_prefixes(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
    repeat: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a new state dict with selected wrapper prefixes removed."""
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
                    if not repeat:
                        break
            if not repeat:
                changed = False
        normalized[new_key] = value
    return normalized


def _candidate_state_dicts(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    """Build plausible key-normalization variants and let the loader choose the best match."""
    wrapper_prefixes = (
        "module.",
        "adapter.",
        "target_anchor_relation_adapter.",
    )
    candidates = [
        ("original", state_dict),
        ("wrapper_stripped", _strip_prefixes(state_dict, wrapper_prefixes, repeat=True)),
        ("model_once_stripped", _strip_prefixes(state_dict, ("model.",), repeat=False)),
        (
            "wrapper_then_model_once",
            _strip_prefixes(_strip_prefixes(state_dict, wrapper_prefixes, repeat=True), ("model.",), repeat=False),
        ),
    ]

    deduped = []
    seen = set()
    for name, candidate in candidates:
        signature = tuple(candidate.keys())
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append((name, candidate))
    return deduped


def _select_best_state_dict(
    model_state_dict: dict[str, torch.Tensor],
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, dict[str, torch.Tensor], int]:
    """Choose the normalization variant with the strongest overlap against the current model keys."""
    target_keys = set(model_state_dict.keys())
    best_name = "original"
    best_state_dict = state_dict
    best_overlap = -1
    best_size = -1

    for name, candidate in _candidate_state_dicts(state_dict):
        overlap = sum(key in target_keys for key in candidate)
        size = len(candidate)
        if overlap > best_overlap or (overlap == best_overlap and size > best_size):
            best_name = name
            best_state_dict = candidate
            best_overlap = overlap
            best_size = size

    return best_name, best_state_dict, best_overlap


def load_target_anchor_relation_adapter(model, adapter_path, strict: bool = False) -> dict[str, Any]:
    """Load a relation adapter checkpoint into a YOLOE model as permissively as possible."""
    adapter_path = Path(adapter_path).expanduser().resolve()
    payload = torch.load(str(adapter_path), map_location="cpu", weights_only=False)
    raw_state_dict = _unwrap_state_dict(payload)
    normalization, state_dict, matching_keys = _select_best_state_dict(model.model.state_dict(), raw_state_dict)

    load_result = model.model.load_state_dict(state_dict, strict=strict)
    payload_dict = payload if isinstance(payload, dict) else {}
    payload_dict.setdefault("path", str(adapter_path))
    payload_dict["normalization"] = normalization
    payload_dict["matching_keys"] = matching_keys
    payload_dict["loaded_state_dict_keys"] = len(state_dict)
    payload_dict["missing_keys"] = list(getattr(load_result, "missing_keys", []))
    payload_dict["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))
    return payload_dict
