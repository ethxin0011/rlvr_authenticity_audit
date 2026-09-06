"""
File: common_split.py
Place in: .../rlvr_authenticity_audit/code/   (alongside 05/08/10/12)

Single source of truth for item ids and the train/test partition.

WHY THIS FILE EXISTS:
Each of 05/08/10/12 previously carried its own copy of get_item_id / is_test_id using Python's
built-in hash(). That hash is randomly salted per process (PYTHONHASHSEED), so:
  * the train/test split differed on every run and every retry;
  * resume never converged (a job logged "RESUMING - 1000 items already written" then
    "pending=997", because those 1000 ids belonged to a different random split);
  * train and test could overlap across scripts -> leakage.

md5 is stable across processes, machines and Python versions. All scripts import from here so
the definitions cannot drift apart again.
"""

import hashlib

QUESTION_KEY = "question"


def stable_hash(s):
    """Process-stable integer hash.

    Do NOT swap this for Python's built-in hash(): it is salted per process via PYTHONHASHSEED
    and returns different values for the same input across runs.
    """
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8], 16)


def get_item_id(row):
    """Stable id for a GooseReason row.

    The official NVIDIA release has no persistent `id` field (schema is
    {question, options, answer}), so we derive one from the question text. Control-set rows
    written by 08 carry an explicit `id`, which is used directly.
    """
    if row.get("id"):
        return row["id"]
    q = row.get(QUESTION_KEY, "") or ""
    return hashlib.md5(q.encode("utf-8")).hexdigest()


def is_test_id(item_id, test_frac=0.2):
    """Deterministic ~80/20 train/test partition, stable across processes and runs."""
    return (stable_hash(item_id) % 1000) < int(test_frac * 1000)
