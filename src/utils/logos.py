from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from dash import get_asset_url

ASSETS_DIR = Path("assets").resolve()
LOGOS_ROOT = (ASSETS_DIR / "logos").resolve()

_LOGO_INDEX: dict[str, Path] | None = None


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _norm_key(s: str) -> str:
    s = _strip_accents(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in LOGOS_ROOT.rglob("*.png"):
        key = _norm_key(p.stem)
        if key not in idx:
            idx[key] = p
    return idx


def find_logo_src(team_name: str) -> str | None:
    global _LOGO_INDEX
    if _LOGO_INDEX is None:
        _LOGO_INDEX = _build_index()

    key = _norm_key(team_name)
    path = _LOGO_INDEX.get(key)
    if not path:
        return None

    rel = path.relative_to(ASSETS_DIR).as_posix()
    return get_asset_url(rel)
