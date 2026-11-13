from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "monthly_budget_usd": 300.0,
    "usd_to_cny": 7.25,
    "include_trae_estimated": True,
    "custom_pricing": {},
}


def get_config_path() -> Path:
    return Path.home() / ".tokenlens" / "config.json"


def load_config() -> dict[str, Any]:
    path = get_config_path()
    cfg = dict(DEFAULT_CONFIG)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:
            pass
    return cfg


def save_config(new_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    for key in ("monthly_budget_usd", "usd_to_cny", "include_trae_estimated", "custom_pricing"):
        if key in new_cfg:
            cfg[key] = new_cfg[key]
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg
