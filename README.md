# Authenticity-Artifact Audit of Synthetic RLVR Data

A two-stage audit of **GooseReason-0.7M**, a synthetic RLVR (reinforcement learning with
verifiable rewards) dataset built by pairing real, human-authored corpus text (the correct
answer) with LLM-generated distractors. This construction introduces an **authenticity
asymmetry** between option classes — real vs. generated text — that is independent of the
intended correctness signal. This repo asks two separate questions about that asymmetry:

1. **Is it detectable?** (Phase 1) — Can a classifier using only surface statistics (no semantic
   content) separate real spans from generated distractors?
2. **Is it exploited?** (Phase 2) — If a policy is trained by RLVR on this data, does it learn to
   rely on the authenticity signal as a shortcut, more than a policy trained on an
   artifact-neutralized control corpus?

These are deliberately kept as separate, sequential questions: detecting an artifact statistically
does not by itself prove a trained model exploits it. See `results/final_report.md` for the
answers this repo found.

## Why this matters

RLVR pipelines increasingly synthesize training data at scale by generating distractors around
real text (e.g. masked-span-and-distractor construction). If the resulting real-vs-generated
asymmetry is both detectable *and* exploitable, models trained on such data could learn a shortcut
disconnected from the reasoning the task is meant to test. This repo provides a reusable,
largely CPU-only protocol to check for that risk *before* committing GPU budget to full-scale
RLVR training, plus the causal-intervention machinery to test exploitation directly when GPU
budget is available.

## Repository structure

```
.
├── README.md                          # this file
│
├── code/                              # all pipeline scripts (submitted as Azure ML jobs)
│   ├── common_split.py                # shared item-id + train/test split + chat-prompt builder
│   │                                   #   (single source of truth — imported by 05/08/10/12)
│   ├── 05_feature_extraction.py       # Phase 1: surface-statistics feature extraction (CPU)
│   ├── 08_construct_control_set.py    # Phase 2: paraphrase-matched control-set construction (GPU)
│   ├── 10_rlvr_training.py            # Phase 2: GRPO/LoRA RLVR training, treatment vs control (GPU)
│   ├── 12_evaluate_models.py          # Phase 2: evaluation — original/neutralized/adversarial (GPU)
│   ├── 14_aggregate_report.py         # combines Phase 1 + Phase 2 into final_report.md + charts
│   └── diagnostic_raw_completions.py  # debugging tool: raw model outputs + adapter hash check
│
├── env/
│   └── environment.yml                # conda spec for the Azure ML custom environment
│
├── notebooks/                         # Azure ML SDK v2 notebooks — orchestration only, no
│   │                                  #   pipeline logic lives here
│   ├── 20_RLVR_workspace_bootstrap.ipynb   # full from-scratch setup: env, data, Phase 1 (optional)
│   ├── 23_phase2_rerun_fixed.ipynb         # Phase 2 control/train/eval submission + aggregation
│   ├── 21_final_aggregation.ipynb          # standalone final-report generation (CPU only)
│   └── 22_diagnostic.ipynb                 # standalone debugging notebook
│
├── data/                              # NOT checked in — populated at runtime
│   ├── raw/                           # downloaded GooseReason-0.7M (math/code/stem .jsonl)
│   ├── control_set/                   # paraphrase-matched control data (train/ and test/)
│   └── features/                      # Phase 1 per-option feature parquet files
│
├── models/                            # NOT checked in — LoRA adapters written here
│   ├── treatment/                     # policy trained on unmodified GooseReason data
│   └── control/                       # policy trained on paraphrase-matched control data
│
├── results/                           # audit outputs
│   ├── classifier_report.json         # Phase 1: AUROC overall + per domain
│   ├── classifier_roc.png
│   ├── eval_report.json               # Phase 2: accuracy + exploitation gap, overall + per domain
│   ├── final_report.md                # combined write-up (this is the paper's data source)
│   ├── exploitation_gap.png
│   └── exploitation_gap_by_domain.png
│
└── paper/                             # arXiv submission source (ACL-style LaTeX)
    ├── main.tex
    ├── tables.tex
    ├── tables_appendix.tex
    ├── references.bib
    ├── acl.sty
    └── figures/
```

## Pipeline overview

| Stage | Script | Compute | What it does |
|---|---|---|---|
| Setup | `03_register_environment.py`, `04_download_and_register_data.py` | CPU instance | Register conda env; download + register GooseReason-0.7M |
| Phase 1 | `05_feature_extraction.py` | CPU cluster | Extract 5 surface-statistics features per option; train classifier (in-notebook) |
| Phase 2 setup | `08_construct_control_set.py` | GPU | Paraphrase (math/stem) or restyle (code) every option to remove authenticity asymmetry, holding option order and task difficulty fixed |
| Phase 2 training | `10_rlvr_training.py` | GPU | GRPO + 4-bit LoRA training of Qwen3-1.7B, run once per arm (treatment / control), matched train-set size |
| Phase 2 eval | `12_evaluate_models.py` | GPU | Evaluate both policies on original / neutralized / adversarial test variants; compute exploitation gap |
| Aggregation | `14_aggregate_report.py` | CPU | Combine both phases into `results/final_report.md` |

## Key design decisions

- **Domain-conditioned control transform**: `code` distractors were found (by manual audit) to be
  minimal single-operator mutations of the gold answer, not freely generated text. A generic
  paraphraser would silently repair the injected bug and destroy task difficulty — so `code` uses
  a restyle-only transform (rename/reformat, explicitly forbidden from changing logic), while
  `math`/`stem` use full semantic paraphrase.
- **Matched training-set size across arms**: `--sample_per_domain` is applied identically to both
  the treatment (raw data) and control (paraphrased data) training pools, so any observed
  difference in exploitation gap cannot be attributed to a difference in training-set size.
- **Stable, process-independent train/test split**: `common_split.py` uses an md5-based hash
  (not Python's built-in `hash()`, which is salted per process) so the partition is identical
  across every script and every run — required for correct resume behavior and to prevent
  train/test leakage.
- **Rule-based, fully verifiable reward**: exact option-letter match, so no second learned
  verifier's bias enters the causal comparison.
- **`enable_thinking=False`**: Qwen3's chat template disables its reasoning preamble at both
  training and evaluation time (applied consistently via `common_split.build_chat_prompt`), so
  short completion-length budgets are not consumed by unparsed reasoning tokens.

## Requirements

- Azure ML workspace with a CPU compute instance, a CPU compute cluster, and a GPU compute
  cluster (single GPU, ≥16GB VRAM tested).
- See `env/environment.yml` for the exact package pins (transformers, trl, peft, accelerate,
  bitsandbytes, datasets — versions verified mutually compatible for Qwen3 + GRPO + QLoRA).

## Citing this work

See `paper/main.tex` for the full write-up, and `results/final_report.md` for the underlying
numbers this repo produced.
