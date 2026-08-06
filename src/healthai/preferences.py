"""Shared preference loading with backwards-compatible key migration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def preferences_path() -> Path:
    configured = os.environ.get("APPLEHEALTH_PREFS")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".applehealth" / "ai_prefs.json"


def _migrate(preferences: dict[str, Any]) -> dict[str, Any]:
    export_path = preferences.get("export_xml") or preferences.get("export_xml_path")
    if export_path:
        preferences["export_xml"] = export_path
        preferences["export_xml_path"] = export_path
    preferences.setdefault("source_mode", "reconcile")
    preferences.setdefault("source_priorities", {})
    preferences.setdefault("source_filter", [])
    preferences.setdefault("unit_system", "metric")
    return preferences


def load_preferences() -> dict[str, Any]:
    path = preferences_path()
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                return _migrate(loaded)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        pass
    return _migrate({})


def save_preferences(preferences: dict[str, Any]) -> None:
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    migrated = _migrate(dict(preferences))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(migrated, handle, indent=2)
