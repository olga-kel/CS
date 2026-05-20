"""Utility helpers for the span-projection reproducibility workflow."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

FUNCTIONAL_UPOS_CORE = {"DET", "AUX", "ADP", "SCONJ", "CCONJ", "PART"}
LEXICAL_UPOS_CORE = {"NOUN", "PROPN", "ADJ", "VERB"}
HIGH_SCOPE_DEPRELS = {"mark", "aux", "cop", "parataxis", "discourse", "cc"}
BOUNDARY_TAGS = {"other", "punct"}


@dataclass
class SpanObservation:
    corpus: str
    sentence_id: Any
    language: str
    span_len: int
    switch_upos: str | None
    switch_deprel: str | None
    switch_form: str | None
    register: str | None = None
    secondary_register: str | None = None
    topic: str | None = None
    sentence_len: int | None = None
    source_row: int | None = None


def parse_listish(value: Any) -> Any:
    """Parse JSON / Python-like list strings robustly."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def normalize_miami_tag(tag: Any) -> str:
    tag = "" if tag is None else str(tag).lower()
    if tag.startswith("eng"):
        return "eng"
    if tag.startswith("spa"):
        return "spa"
    return "other"


def normalize_paraguay_tag(tag: Any) -> str:
    tag = "" if tag is None else str(tag).lower()
    if tag.startswith("gn"):
        return "gua"
    if tag.startswith("es") or tag.startswith("ne"):
        return "spa"
    return "other"


def is_functional_upos(upos: Any, corpus: str | None = None) -> bool:
    upos = "" if upos is None else str(upos).upper()
    return upos in FUNCTIONAL_UPOS_CORE


def classify_upos(upos: Any) -> str:
    upos = "" if upos is None else str(upos).upper()
    if upos in FUNCTIONAL_UPOS_CORE:
        return "functional"
    if upos in LEXICAL_UPOS_CORE:
        return "lexical"
    return "other"


def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return float("nan")
    return (x.mean() - y.mean()) / np.sqrt(pooled)


def summarise_numeric(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[value_col]
        .agg(N="count", mean="mean", sd="std", median="median", min="min", max="max")
        .reset_index()
    )
    return out[group_cols + ["N", "mean", "sd", "median", "min", "max"]]


def ttest_by_group(
    df: pd.DataFrame,
    corpus_col: str = "corpus",
    group_col: str = "group",
    value_col: str = "span_len",
) -> pd.DataFrame:
    rows = []
    for corpus, g in df.groupby(corpus_col):
        func = g.loc[g[group_col] == "functional", value_col].dropna().astype(float)
        lex = g.loc[g[group_col] == "lexical", value_col].dropna().astype(float)
        if len(func) >= 2 and len(lex) >= 2:
            t_stat, p_val = stats.ttest_ind(func, lex, equal_var=False)
        else:
            t_stat, p_val = (float("nan"), float("nan"))
        rows.append(
            {
                "corpus": corpus,
                "t": t_stat,
                "p": p_val,
                "cohens_d": cohens_d(func, lex),
                "n_func": len(func),
                "n_lex": len(lex),
            }
        )
    return pd.DataFrame(rows)
