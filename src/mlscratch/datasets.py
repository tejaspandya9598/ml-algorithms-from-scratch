"""Local copies of the UCI datasets the notebooks use, so they run without the network.

`load_uci(id)` mirrors `ucimlrepo.fetch_ucirepo(id=...)` closely enough for the
notebooks (`.data.features`, `.data.targets`). It reads `data/uci/<id>_*.csv` and only
falls back to the UCI API if a copy is missing. The datasets are CC BY 4.0:

    17   Breast Cancer Wisconsin (Diagnostic)
    19   Car Evaluation
    186  Wine Quality
    222  Bank Marketing
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

UCI_DIR = Path(__file__).resolve().parents[2] / "data" / "uci"


def load_uci(id: int) -> SimpleNamespace:  # noqa: A002 - same keyword as fetch_ucirepo
    feats, targs = UCI_DIR / f"{id}_features.csv", UCI_DIR / f"{id}_targets.csv"
    if feats.exists() and targs.exists():
        return SimpleNamespace(data=SimpleNamespace(features=pd.read_csv(feats),
                                                    targets=pd.read_csv(targs)))
    from ucimlrepo import fetch_ucirepo  # only needed when a copy is missing

    return fetch_ucirepo(id=id)
