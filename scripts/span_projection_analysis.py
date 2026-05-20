#!/usr/bin/env python3
"""Reproduce the span-projection analysis for the Miami and Paraguay corpora.

The script reads the processed corpora, applies an explicit Paraguayan loanword
relabeling step if a mapping file is provided, reconstructs token- and
sentence-level switch statistics, and regenerates the summary tables used in the
manuscript.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from span_projection_utils import (
    FUNCTIONAL_UPOS_CORE,
    HIGH_SCOPE_DEPRELS,
    classify_upos,
    is_functional_upos,
    normalize_miami_tag,
    normalize_paraguay_tag,
    parse_listish,
    safe_int,
    summarise_numeric,
    ttest_by_group,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIAMI = PACKAGE_ROOT / "data" / "miami_gpt_labels_sent.xlsx"
DEFAULT_PARAGUAY = PACKAGE_ROOT / "data" / "spa_gua_gpt_labels_sent-2.xlsx"
DEFAULT_LOANWORDS = PACKAGE_ROOT / "data" / "paraguay_loanword_mapping.csv"
DEFAULT_OUTDIR = PACKAGE_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miami", type=Path, default=DEFAULT_MIAMI)
    parser.add_argument("--paraguay", type=Path, default=DEFAULT_PARAGUAY)
    parser.add_argument("--loanword-map", type=Path, default=DEFAULT_LOANWORDS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def token_lookup_miami(row: pd.Series) -> dict[int, dict[str, Any]]:
    tokens = parse_listish(row["tokens_UD"]) or []
    lookup: dict[int, dict[str, Any]] = {}
    for tok in tokens:
        idx = safe_int(tok.get("token_info", {}).get("token_index"))
        if idx is not None:
            lookup[idx] = tok
    return lookup


def load_loanword_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "form" not in df.columns or "relabel_to" not in df.columns:
        return {}
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        form = str(row["form"]).strip().lower()
        relabel_to = str(row["relabel_to"]).strip().lower()
        if form and relabel_to:
            mapping[form] = relabel_to
    return mapping


def apply_paraguay_loanword_map(ud_tags: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    if not mapping:
        return ud_tags
    out: list[dict[str, Any]] = []
    for tok in ud_tags:
        tok = dict(tok)
        form = str(tok.get("FORM", "")).strip().lower()
        if form in mapping and str(tok.get("lang_tag", "")).lower().startswith("es"):
            relabel = mapping[form]
            if relabel.startswith("gua"):
                tok["lang_tag"] = "gn"
            elif relabel.startswith("spa"):
                tok["lang_tag"] = "es"
        out.append(tok)
    return out


def iter_runs_from_sequence(seq: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_lang: str | None = None
    for item in seq:
        if item.get("boundary"):
            if current:
                runs.append(current)
            current = []
            current_lang = None
            continue
        lang = item["lang"]
        if current_lang is None:
            current_lang = lang
            current = [item]
        elif lang == current_lang:
            current.append(item)
        else:
            runs.append(current)
            current = [item]
            current_lang = lang
    if current:
        runs.append(current)
    return runs


def explode_miami(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        tokens = parse_listish(row["tokens_UD"]) or []
        seq: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        for tok in tokens:
            info = tok.get("token_info", {}) if isinstance(tok, dict) else {}
            ud = tok.get("UD_tag", {}) if isinstance(tok, dict) else {}
            lang = normalize_miami_tag(info.get("lang_tag"))
            idx = safe_int(info.get("token_index"))
            if lang == "other":
                seq.append({"boundary": True})
            else:
                seq.append(
                    {
                        "boundary": False,
                        "lang": lang,
                        "upos": ud.get("upos"),
                        "deprel": ud.get("deprel"),
                        "form": ud.get("form") or info.get("token"),
                        "token_id": idx,
                        "raw_lang": info.get("lang_tag"),
                    }
                )
                token_rows.append(
                    {
                        "corpus": "Miami",
                        "sentence_id": row.get("sent_id"),
                        "speaker": row.get("speaker"),
                        "register": row.get("situation"),
                        "secondary_register": row.get("secondary_loc"),
                        "topic": row.get("topic"),
                        "sentence_len": row.get("sen_len"),
                        "token_id": idx,
                        "form": ud.get("form") or info.get("token"),
                        "upos": ud.get("upos"),
                        "deprel": ud.get("deprel"),
                        "language": "English" if lang == "eng" else "Spanish",
                        "raw_lang_tag": info.get("lang_tag"),
                        "is_functional": is_functional_upos(ud.get("upos")),
                        "upos_class": classify_upos(ud.get("upos")),
                        "high_scope_deprel": str(ud.get("deprel")) in HIGH_SCOPE_DEPRELS,
                    }
                )

        runs = iter_runs_from_sequence(seq)
        for run_idx, run in enumerate(runs):
            if run_idx == 0:
                continue
            first = run[0]
            rows.append(
                {
                    "corpus": "Miami",
                    "source_row": i,
                    "sentence_id": row.get("sent_id"),
                    "language": "English" if first["lang"] == "eng" else "Spanish",
                    "span_len": len(run),
                    "switch_form": first.get("form"),
                    "switch_upos": first.get("upos"),
                    "switch_deprel": first.get("deprel"),
                    "switch_lang_tag": first.get("raw_lang"),
                    "register": row.get("situation"),
                    "secondary_register": row.get("secondary_loc"),
                    "topic": row.get("topic"),
                    "sentence_len": row.get("sen_len"),
                    "speaker": row.get("speaker"),
                    "age": row.get("age"),
                    "gender": row.get("gender"),
                    "is_functional": is_functional_upos(first.get("upos")),
                    "upos_class": classify_upos(first.get("upos")),
                    "high_scope_deprel": str(first.get("deprel")) in HIGH_SCOPE_DEPRELS,
                    "switch_start": 1,
                }
            )
        for tr in token_rows:
            tr["switch_start"] = 0
        # Add switch-start tokens from the extracted spans.
        run_start_positions = []
        running_index = 0
        for run_idx, run in enumerate(runs):
            for j, tok in enumerate(run):
                if run_idx > 0 and j == 0:
                    run_start_positions.append(tok.get("token_id"))
        for tr in token_rows:
            if tr["token_id"] in run_start_positions:
                tr["switch_start"] = 1
        rows.extend(token_rows)
    return pd.DataFrame(rows)


def explode_paraguay(df: pd.DataFrame, loanword_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        ud_tags = parse_listish(row["UD_tags"]) or []
        ud_tags = apply_paraguay_loanword_map(ud_tags, loanword_map)

        seq: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        for tok in ud_tags:
            lang = normalize_paraguay_tag(tok.get("lang_tag"))
            if lang == "other":
                seq.append({"boundary": True})
                continue
            seq.append(
                {
                    "boundary": False,
                    "lang": lang,
                    "upos": tok.get("UPOS"),
                    "deprel": tok.get("DEPREL"),
                    "form": tok.get("FORM"),
                    "token_id": safe_int(tok.get("ID")),
                    "raw_lang": tok.get("lang_tag"),
                }
            )
            token_rows.append(
                {
                    "corpus": "Paraguay",
                    "sentence_id": row.get("sent_id"),
                    "register": row.get("Formality"),
                    "secondary_register": row.get("Genre"),
                    "topic": row.get("Topic"),
                    "sentence_len": row.get("sentence_len"),
                    "token_id": safe_int(tok.get("ID")),
                    "form": tok.get("FORM"),
                    "upos": tok.get("UPOS"),
                    "deprel": tok.get("DEPREL"),
                    "language": "Guarani" if lang == "gua" else "Spanish",
                    "raw_lang_tag": tok.get("lang_tag"),
                    "is_functional": is_functional_upos(tok.get("UPOS")),
                    "upos_class": classify_upos(tok.get("UPOS")),
                    "high_scope_deprel": str(tok.get("DEPREL")) in HIGH_SCOPE_DEPRELS,
                }
            )

        runs = iter_runs_from_sequence(seq)
        for run_idx, run in enumerate(runs):
            if run_idx == 0:
                continue
            first = run[0]
            rows.append(
                {
                    "corpus": "Paraguay",
                    "source_row": i,
                    "sentence_id": row.get("sent_id"),
                    "language": "Guarani" if first["lang"] == "gua" else "Spanish",
                    "span_len": len(run),
                    "switch_form": first.get("form"),
                    "switch_upos": first.get("upos"),
                    "switch_deprel": first.get("deprel"),
                    "switch_lang_tag": first.get("raw_lang"),
                    "register": row.get("Formality"),
                    "secondary_register": row.get("Genre"),
                    "topic": row.get("Topic"),
                    "sentence_len": row.get("sentence_len"),
                    "has_emoji": row.get("has_emoji"),
                    "total_spa": row.get("total_spa"),
                    "total_gua": row.get("total_gua"),
                    "is_functional": is_functional_upos(first.get("upos")),
                    "upos_class": classify_upos(first.get("upos")),
                    "high_scope_deprel": str(first.get("deprel")) in HIGH_SCOPE_DEPRELS,
                    "switch_start": 1,
                }
            )
        for tr in token_rows:
            tr["switch_start"] = 0
        run_start_positions = []
        for run_idx, run in enumerate(runs):
            for j, tok in enumerate(run):
                if run_idx > 0 and j == 0:
                    run_start_positions.append(tok.get("token_id"))
        for tr in token_rows:
            if tr["token_id"] in run_start_positions:
                tr["switch_start"] = 1
        rows.extend(token_rows)
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sample_for_logit(df: pd.DataFrame, max_rows: int = 8000, random_state: int = 42) -> pd.DataFrame:
    """Create a balanced sample for the token-level logistic model."""
    if len(df) <= max_rows:
        return df.copy()
    positives = df[df["switch_start"] == 1]
    negatives = df[df["switch_start"] == 0]
    n_pos = min(len(positives), max_rows // 2)
    n_neg = min(len(negatives), max_rows - n_pos)
    pos_sample = positives.sample(n=n_pos, random_state=random_state) if n_pos else positives.iloc[0:0]
    neg_sample = negatives.sample(n=n_neg, random_state=random_state) if n_neg else negatives.iloc[0:0]
    return pd.concat([pos_sample, neg_sample], ignore_index=True)


def regression_table(model, model_name: str, corpus: str) -> pd.DataFrame:
    out = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    out.insert(0, "corpus", corpus)
    out.insert(1, "model", model_name)
    return out


def safe_glm(formula: str, df: pd.DataFrame, family, cov_type: str | None = None, groups=None):
    fit = smf.glm(formula, data=df, family=family).fit()
    if cov_type:
        kw = {}
        if cov_type == "cluster" and groups is not None:
            kw = {"groups": groups}
        fit = fit.get_robustcov_results(cov_type=cov_type, **kw)
    return fit


def run_regressions(miami_long: pd.DataFrame, paraguay_long: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    notes: list[str] = []
    results: list[pd.DataFrame] = []
    extras: dict[str, pd.DataFrame] = {}

    # Sentence-level counts for Poisson models.
    mi_switch_counts = miami_long.groupby("sentence_id").agg(
        switch_count=("switch_start", "sum"),
        sentence_len=("sentence_len", "first"),
        register=("register", "first"),
        topic=("topic", "first"),
        speaker=("speaker", "first"),
    ).reset_index()
    pa_switch_counts = paraguay_long.groupby("sentence_id").agg(
        switch_count=("switch_start", "sum"),
        sentence_len=("sentence_len", "first"),
        register=("register", "first"),
        topic=("topic", "first"),
        secondary_register=("secondary_register", "first"),
    ).reset_index()

    # Sentence-level mean span lengths.
    mi_sentence_spans = miami_long[miami_long["switch_start"] == 1].groupby(["sentence_id", "register", "topic"]).agg(
        mean_span_len=("span_len", "mean"),
        sentence_len=("sentence_len", "first"),
        n_switches=("span_len", "count"),
    ).reset_index()
    pa_sentence_spans = paraguay_long[paraguay_long["switch_start"] == 1].groupby(["sentence_id", "register", "secondary_register", "topic"]).agg(
        mean_span_len=("span_len", "mean"),
        sentence_len=("sentence_len", "first"),
        n_switches=("span_len", "count"),
    ).reset_index()

    # Poisson: sentence-level switch counts.
    mi_pois = smf.glm("switch_count ~ C(register) + sentence_len", data=mi_switch_counts, family=sm.families.Poisson()).fit()
    results.append(regression_table(mi_pois, "Poisson_switch_count", "Miami"))
    notes.append(f"Miami Poisson: n={int(mi_switch_counts.shape[0])}, AIC={mi_pois.aic:.2f}")

    pa_pois = smf.glm("switch_count ~ C(register) + C(secondary_register) + sentence_len", data=pa_switch_counts, family=sm.families.Poisson()).fit()
    results.append(regression_table(pa_pois, "Poisson_switch_count", "Paraguay"))
    notes.append(f"Paraguay Poisson: n={int(pa_switch_counts.shape[0])}, AIC={pa_pois.aic:.2f}")

    # OLS: mean span length.
    mi_ols = smf.ols("mean_span_len ~ C(register) + sentence_len", data=mi_sentence_spans).fit(cov_type="HC3")
    results.append(regression_table(mi_ols, "OLS_mean_span", "Miami"))
    notes.append(f"Miami OLS mean span: n={int(mi_sentence_spans.shape[0])}, R2={mi_ols.rsquared:.3f}")

    pa_ols = smf.ols("mean_span_len ~ C(register) + C(secondary_register) + sentence_len", data=pa_sentence_spans).fit(cov_type="HC3")
    results.append(regression_table(pa_ols, "OLS_mean_span", "Paraguay"))
    notes.append(f"Paraguay OLS mean span: n={int(pa_sentence_spans.shape[0])}, R2={pa_ols.rsquared:.3f}")

    # Token-level logistic regression: whether a token starts a switched span.
    mi_tok = sample_for_logit(miami_long.dropna(subset=["upos", "deprel", "register", "sentence_len"]).copy())
    pa_tok = sample_for_logit(paraguay_long.dropna(subset=["upos", "deprel", "register", "sentence_len"]).copy())

    # Keep a balanced set of predictors that are present in both corpora.
    mi_logit = smf.glm(
        "switch_start ~ C(register) + C(language) + C(upos_class) + sentence_len",
        data=mi_tok,
        family=sm.families.Binomial(),
    ).fit(maxiter=100, disp=False)
    results.append(regression_table(mi_logit, "Logit_switch_start", "Miami"))
    notes.append(f"Miami logistic: n={int(mi_tok.shape[0])}, pseudo-R2 not reported")

    pa_logit = smf.glm(
        "switch_start ~ C(register) + C(language) + C(upos_class) + sentence_len",
        data=pa_tok,
        family=sm.families.Binomial(),
    ).fit(maxiter=100, disp=False)
    results.append(regression_table(pa_logit, "Logit_switch_start", "Paraguay"))
    notes.append(f"Paraguay logistic: n={int(pa_tok.shape[0])}, pseudo-R2 not reported")

    # Cross-corpus interaction model.
    combined = pd.concat(
        [
            mi_sentence_spans.assign(corpus="Miami", language="English"),
            pa_sentence_spans.assign(corpus="Paraguay", language="Spanish"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined["is_functional_num"] = np.nan
    combined_model = smf.ols("mean_span_len ~ C(corpus) * C(register) + sentence_len", data=combined).fit(cov_type="HC3")
    results.append(regression_table(combined_model, "OLS_cross_corpus", "Combined"))
    notes.append(f"Combined interaction model: n={int(combined.shape[0])}, R2={combined_model.rsquared:.3f}")

    extras["mi_switch_counts"] = mi_switch_counts
    extras["pa_switch_counts"] = pa_switch_counts
    extras["mi_sentence_spans"] = mi_sentence_spans
    extras["pa_sentence_spans"] = pa_sentence_spans

    return pd.concat(results, ignore_index=True), notes, extras


def sentence_level_summary(long_df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    sent = (
        long_df[long_df["switch_start"] == 1]
        .groupby("sentence_id")
        .agg(
            n_switches=("switch_start", "sum"),
            mean_span_len=("span_len", "mean"),
            median_span_len=("span_len", "median"),
            sentence_len=("sentence_len", "first"),
            register=("register", "first"),
            topic=("topic", "first"),
        )
        .reset_index()
    )
    sent.insert(0, "corpus", corpus)
    return sent


def make_key_results(table2: pd.DataFrame, ttests: pd.DataFrame, reg_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in table2.iterrows():
        rows.append(
            {
                "result_type": "Functional vs Lexical",
                "corpus": r["corpus"],
                "value": f"mean functional={r.loc['mean'] if 'mean' in table2.columns else ''}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    miami_raw = pd.read_excel(args.miami)
    paraguay_raw = pd.read_excel(args.paraguay)
    loanword_map = load_loanword_map(args.loanword_map)

    miami_long = explode_miami(miami_raw)
    paraguay_long = explode_paraguay(paraguay_raw, loanword_map)
    combined = pd.concat([miami_long, paraguay_long], ignore_index=True)

    combined_spans = combined[combined["switch_start"] == 1].copy()
    func_lex = combined_spans[combined_spans["upos_class"].isin(["functional", "lexical"])][["corpus", "upos_class", "span_len"]].rename(columns={"upos_class": "group"})
    table2 = summarise_numeric(func_lex, ["corpus", "group"], "span_len").sort_values(["corpus", "group"]).reset_index(drop=True)
    ttests = ttest_by_group(func_lex, corpus_col="corpus", group_col="group", value_col="span_len")
    upos_table = summarise_numeric(combined_spans.dropna(subset=["switch_upos"]), ["corpus", "switch_upos"], "span_len").rename(columns={"switch_upos": "upos"})
    deprel_table = summarise_numeric(combined_spans.dropna(subset=["switch_deprel"]), ["corpus", "switch_deprel"], "span_len").rename(columns={"switch_deprel": "deprel"})

    # Sentence-level summaries used by the extended models.
    mi_sent = sentence_level_summary(miami_long, "Miami")
    pa_sent = sentence_level_summary(paraguay_long, "Paraguay")

    reg_table, notes, extras = run_regressions(miami_long, paraguay_long)

    # Write output files.
    write_table(miami_long, args.outdir / "miami_span_observations.csv")
    write_table(paraguay_long, args.outdir / "paraguay_span_observations.csv")
    write_table(combined, args.outdir / "combined_span_observations.csv")
    write_table(mi_sent, args.outdir / "miami_sentence_level_summary.csv")
    write_table(pa_sent, args.outdir / "paraguay_sentence_level_summary.csv")

    write_table(table2, args.outdir / "TABLE2_functional_vs_lexical.csv")
    write_table(ttests, args.outdir / "TableS2_ttests_and_effectsizes.csv")
    write_table(upos_table, args.outdir / "TableS1_by_UPOS.csv")
    write_table(deprel_table, args.outdir / "SUPPLEMENT_Full_DEPREL_Table.csv")
    write_table(reg_table, args.outdir / "regression_terms.csv")

    # Key result mapping file to make reviewer navigation easier.
    key_results = pd.DataFrame(
        [
            {"paper_table": "Table 5", "output_file": "regression_terms.csv", "content": "Expanded regression terms for Poisson, OLS, and logistic models"},
            {"paper_table": "Table 2", "output_file": "TABLE2_functional_vs_lexical.csv", "content": "Functional vs. lexical span-length summary"},
            {"paper_table": "Table S1", "output_file": "TableS1_by_UPOS.csv", "content": "Span summary by UPOS"},
            {"paper_table": "Table S2", "output_file": "TableS2_ttests_and_effectsizes.csv", "content": "T-tests and effect sizes"},
            {"paper_table": "Supplementary DEPREL table", "output_file": "SUPPLEMENT_Full_DEPREL_Table.csv", "content": "Span summary by DEPREL"},
        ]
    )
    write_table(key_results, args.outdir / "paper_to_output_map.csv")

    report = [
        "Span-projection reproducibility run",
        f"Miami observations (span rows): {len(miami_long)}",
        f"Paraguay observations (span rows): {len(paraguay_long)}",
        f"Combined observations (span rows): {len(combined)}",
        f"Loanword map entries loaded: {len(loanword_map)}",
        "",
        "Functional UPOS core: " + ", ".join(sorted(FUNCTIONAL_UPOS_CORE)),
        "Lexical UPOS core: NOUN, PROPN, ADJ, VERB",
        "",
        "Model notes:",
        *[f"  - {x}" for x in notes],
        "",
        "Sentence-level switch counts are derived from contiguous same-language runs in the token-level data.",
        "Token-level switch_start = 1 marks the first token of each switched span after the initial span.",
    ]
    (args.outdir / "analysis_report.txt").write_text("\n".join(report), encoding="utf-8")

    manifest = [
        "Files in this package:",
        "- data/miami_gpt_labels_sent.xlsx",
        "- data/spa_gua_gpt_labels_sent-2.xlsx",
        "- data/paraguay_loanword_mapping.csv",
        "- scripts/span_projection_analysis.py",
        "- scripts/span_projection_utils.py",
        "- results/*.csv ( tables and model outputs)",
        "- README.md",
        "- requirements.txt",
    ]
    (args.outdir / "MANIFEST.txt").write_text("\n".join(manifest), encoding="utf-8")

    print(f"Wrote reproducibility outputs to: {args.outdir}")
    print(f"Miami spans: {len(miami_long)}; Paraguay spans: {len(paraguay_long)}; Combined: {len(combined)}")


if __name__ == "__main__":
    main()
