"""
File: 05_feature_extraction.py
Run as: AML job on cpuclusterrlvr (Phase 1 — optional if you already have Phase 1 results).

Computes per-option surface-statistics features to test whether "real corpus span" vs
"LLM-generated distractor" is separable from surface statistics alone.

Schema handled: {"question": str, "options": [str], "answer": "<letter>"} where `answer` is the
POSITION letter of the correct option, not its text.

Notes:
  - id helper imported from common_split.py. This script never filters by train/test — it
    features every sampled item — so Phase 1 results were unaffected by the old hash() bug.
  - kenlm_ppl removed (kenlm cannot build on the ACPT image; the column was always NaN).

Output: {domain}_features.parquet + combined_features.parquet, one row per option, columns:
        item_id, domain, option_idx, label (1=gold, 0=distractor), text, <features>
"""

import os
import json
import argparse
import random
import math
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from common_split import get_item_id

QUESTION_KEY = "question"
OPTIONS_KEY = "options"
ANSWER_KEY = "answer"
RARE_ZIPF_THRESHOLD = 3.0

_NLTK_READY = False
_REF_POS_DIST = None


def get_gold_and_distractors(row):
    options = row.get(OPTIONS_KEY, [])
    answer_letter = row.get(ANSWER_KEY, "")
    if not options or not answer_letter:
        return None, []
    try:
        idx = ord(str(answer_letter).strip().upper()[0]) - ord("A")
    except Exception:
        return None, []
    if idx < 0 or idx >= len(options):
        return None, []
    return options[idx], [opt for i, opt in enumerate(options) if i != idx]


def _try_load_nltk():
    global _NLTK_READY, _REF_POS_DIST
    try:
        import nltk
        for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                    "averaged_perceptron_tagger_eng", "brown"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        from nltk.corpus import brown
        from nltk import pos_tag
        tags = [t for _, t in pos_tag(brown.words()[:200000])]
        c = Counter(tags)
        total = sum(c.values())
        _REF_POS_DIST = {k: v / total for k, v in c.items()}
        _NLTK_READY = True
    except Exception as e:
        print(f"[worker] NLTK unavailable ({e}); pos_kl_div will be NaN.", flush=True)
        _NLTK_READY = False


def _worker_init():
    _try_load_nltk()


def _char_entropy(text):
    if not text:
        return np.nan
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _burstiness(lengths):
    if len(lengths) < 2:
        return 0.0
    arr = np.array(lengths, dtype=float)
    mean = arr.mean()
    return 0.0 if mean == 0 else float(arr.var() / mean)


def _rare_token_rate(tokens):
    if not tokens:
        return np.nan
    rare = sum(1 for t in tokens if zipf_frequency(t.lower(), "en") < RARE_ZIPF_THRESHOLD)
    return rare / len(tokens)


def _pos_kl_div(text):
    if not _NLTK_READY:
        return np.nan
    try:
        from nltk import word_tokenize, pos_tag
        tokens = word_tokenize(text)
        if not tokens:
            return np.nan
        c = Counter(t for _, t in pos_tag(tokens))
        total = sum(c.values())
        dist = {k: v / total for k, v in c.items()}
        eps = 1e-6
        return float(sum(
            dist.get(k, eps) * math.log((dist.get(k, eps) + eps) / (_REF_POS_DIST.get(k, eps) + eps))
            for k in set(dist) | set(_REF_POS_DIST)
        ))
    except Exception:
        return np.nan


def extract_features(text):
    words = text.split()
    sents = [s.strip() for s in text.split(".") if s.strip()]
    return {
        "rare_token_rate": _rare_token_rate(words),
        "word_len_burstiness": _burstiness([len(w) for w in words]),
        "sent_len_burstiness": _burstiness([len(s) for s in sents] if sents else [len(text)]),
        "char_entropy": _char_entropy(text),
        "pos_kl_div": _pos_kl_div(text),
    }


def process_item(args):
    idx, row, domain = args
    item_id = get_item_id(row)
    gold, distractors = get_gold_and_distractors(row)
    out = []
    if isinstance(gold, str) and gold.strip():
        out.append({"item_id": item_id, "domain": domain, "option_idx": 0,
                     "label": 1, "text": gold, **extract_features(gold)})
    for i, d in enumerate(distractors):
        if isinstance(d, str) and d.strip():
            out.append({"item_id": item_id, "domain": domain, "option_idx": i + 1,
                         "label": 0, "text": d, **extract_features(d)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--sample_per_domain", type=int, default=15000)
    p.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 1))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    frames = []
    for domain in ["math", "code", "stem"]:
        path = os.path.join(args.input_dir, f"{domain}.jsonl")
        if not os.path.exists(path):
            print(f"Skipping missing split: {path}", flush=True)
            continue

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        if len(rows) > args.sample_per_domain:
            rows = random.sample(rows, args.sample_per_domain)

        print(f"[{domain}] {len(rows)} items, {args.num_workers} workers ...", flush=True)
        work = [(i, r, domain) for i, r in enumerate(rows)]
        with Pool(processes=args.num_workers, initializer=_worker_init) as pool:
            results = pool.map(process_item, work, chunksize=25)

        df = pd.DataFrame([r for sub in results for r in sub])
        if df.empty:
            print(f"[{domain}] WARNING: no rows extracted — check schema!", flush=True)
        else:
            print(f"[{domain}] label balance:\n{df['label'].value_counts()}", flush=True)
        out_path = os.path.join(args.output_dir, f"{domain}_features.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[{domain}] saved {len(df)} rows -> {out_path}", flush=True)
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        cp = os.path.join(args.output_dir, "combined_features.parquet")
        combined.to_parquet(cp, index=False)
        print(f"Saved combined: {len(combined)} rows -> {cp}", flush=True)
        print(f"Overall label balance:\n{combined['label'].value_counts()}", flush=True)


if __name__ == "__main__":
    main()
