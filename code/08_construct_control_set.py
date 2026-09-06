"""
File: 08_construct_control_set.py
Run as: AML job on rlvr2 (GPU) — run twice: --split train and --split test.

Builds the artifact-neutralized CONTROL dataset: every option (gold + all distractors) passes
through the SAME transform, removing the "real corpus text vs LLM-generated text" asymmetry
while option ORDER is preserved so the `answer` letter stays valid.

Domain-conditioned transform (deliberate):
  math / stem : full semantic paraphrase.
  code        : RESTYLE-ONLY (rename vars / reformat / recomment) with an explicit instruction
                NOT to fix any bug. A spot-check showed `code` distractors are minimal-mutation
                bug injections; a normal paraphrase would silently repair them and destroy what
                makes the MCQ solvable.

Resilience (all needed on spot/LowPriority VMs):
  * incremental per-chunk writes, flushed + fsync'd to the rw_mount blob path;
  * resume: on restart, already-written ids are read back and skipped;
  * deterministic ordering, so the sample and resume point are stable across restarts;
  * graceful SIGTERM -> finishes the in-flight chunk, exits 1 so AML retries and resumes;
  * flushed progress logs (Python buffers stdout when not on a TTY).

Memory safety:
  * token-budget batching: batch_size * longest_seq_in_batch <= --batch_token_budget, with a
    --max_seqs_cap ceiling. This decouples GPU batch size from checkpoint granularity; an
    earlier version conflated them and OOM'd after 18h when it hit a run of 8-option items
    (a [batch x seq_len x 152k vocab] logits tensor cast to fp32 is several GB).
  * adaptive OOM backoff: halves the sub-batch down to 1; a single pathological option falls
    back to its original text rather than killing a multi-hour run.
  * expandable_segments to limit fragmentation.

Throughput:
  * length-sorted packing (minimal padding), dedup cache for repeated option strings, and a
    decode budget scaled to input length (bounded above by --max_new_tokens).
"""

import os

# MUST precede torch's CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import random
import argparse
import signal
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common_split import get_item_id, is_test_id

QUESTION_KEY = "question"
OPTIONS_KEY = "options"
ANSWER_KEY = "answer"

PARAPHRASE_PROMPTS = {
    "prose": (
        "Paraphrase the following text. Preserve its exact meaning and approximate length "
        "and style. Output ONLY the paraphrased text with no preamble, quotes, or explanation.\n\n"
        "Text: {text}\n\nParaphrase:"
    ),
    "code": (
        "Rewrite the following code snippet in a different STYLE only: rename variables, adjust "
        "whitespace/formatting, and rephrase or remove comments as needed. You MUST preserve the "
        "exact logic, control flow, operators, and behavior — do NOT fix, change, or "
        "'improve' any bug or logic error present in the snippet, even if it looks incorrect. "
        "Output ONLY the restyled code with no preamble, quotes, or explanation.\n\n"
        "Code: {text}\n\nRestyled code:"
    ),
}

DOMAIN_TO_PROMPT_STYLE = {"math": "prose", "stem": "prose", "code": "code"}

_SHUTDOWN_REQUESTED = False


def _handle_sigterm(signum, frame):
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    print("\n[signal] SIGTERM received (likely spot-VM preemption). Finishing current chunk, "
          "then exiting cleanly. Progress is checkpointed — the retry will resume.", flush=True)


def load_completed_ids(out_path):
    """Ids already written. Tolerates a truncated final line from an abrupt kill."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("id"):
                done.add(obj["id"])
    return done


def load_model(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set here so every call path gets correct decoder-only padding.
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, text, prompt_style):
    user_msg = PARAPHRASE_PROMPTS[prompt_style].format(text=text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _generate_once(model, tokenizer, chat_prompts, max_new_tokens, max_input_len):
    inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_input_len).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [d.strip() for d in decoded]


def pack_batches(lengths, order, token_budget, max_seqs_cap):
    """Greedily pack length-sorted indices so batch_size * longest_in_batch <= token_budget."""
    batch, batch_max = [], 0
    for idx in order:
        cand_max = max(batch_max, lengths[idx])
        if batch and ((len(batch) + 1) * cand_max > token_budget or len(batch) + 1 > max_seqs_cap):
            yield batch
            batch, batch_max = [idx], lengths[idx]
        else:
            batch.append(idx)
            batch_max = cand_max
    if batch:
        yield batch


def paraphrase_texts(model, tokenizer, texts, prompt_style, args, cache):
    """OOM-safe, throughput-optimised. Returns a list aligned with `texts`."""
    n = len(texts)
    results = [None] * n

    todo = []
    if args.dedup:
        for i, t in enumerate(texts):
            c = cache.get(t)
            if c is not None:
                results[i] = c
            else:
                todo.append(i)
    else:
        todo = list(range(n))

    unique_map = {}
    for i in todo:
        unique_map.setdefault(texts[i], []).append(i)
    unique_texts = list(unique_map.keys())
    if not unique_texts:
        return [r if r is not None else texts[k] for k, r in enumerate(results)]

    chat_prompts = [build_chat_prompt(tokenizer, t, prompt_style) for t in unique_texts]
    lengths = [len(tokenizer(p, truncation=True, max_length=args.max_input_len)["input_ids"])
               for p in chat_prompts]
    order = sorted(range(len(unique_texts)), key=lambda k: lengths[k])

    for batch_idxs in pack_batches(lengths, order, args.batch_token_budget, args.max_seqs_cap):
        pending = list(batch_idxs)
        while pending:
            try_idxs = pending
            while True:
                sub = [chat_prompts[k] for k in try_idxs]
                longest = max(lengths[k] for k in try_idxs)
                dyn = min(args.max_new_tokens, int(math.ceil(1.6 * longest)) + 32)
                try:
                    decoded = _generate_once(model, tokenizer, sub, dyn, args.max_input_len)
                    for k, txt in zip(try_idxs, decoded):
                        src = unique_texts[k]
                        if args.dedup:
                            cache[src] = txt
                        for i in unique_map[src]:
                            results[i] = txt
                    pending = pending[len(try_idxs):]
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(try_idxs) > 1:
                        new_size = max(1, len(try_idxs) // 2)
                        print(f"[oom] CUDA OOM — halving sub-batch {len(try_idxs)} -> {new_size}",
                              flush=True)
                        try_idxs = try_idxs[:new_size]
                        continue
                    k = try_idxs[0]
                    src = unique_texts[k]
                    prev = (src[:80] + "...") if len(src) > 80 else src
                    print(f"[oom] Single-sequence OOM — keeping ORIGINAL text. Preview: {prev!r}",
                          flush=True)
                    if args.dedup:
                        cache[src] = src
                    for i in unique_map[src]:
                        results[i] = src
                    pending = pending[1:]
                    break

    return [r if r is not None else texts[k] for k, r in enumerate(results)]


def process_domain(model, tokenizer, domain, rows, out_path, args):
    prompt_style = DOMAIN_TO_PROMPT_STYLE.get(domain, "prose")

    completed = load_completed_ids(out_path)
    if completed:
        print(f"[{domain}] RESUMING — {len(completed)} items already written.", flush=True)

    pending = [r for r in rows if r["_item_id"] not in completed]
    print(f"[{domain}] style='{prompt_style}' | total={len(rows)} done={len(completed)} "
          f"pending={len(pending)}", flush=True)
    if not pending:
        print(f"[{domain}] nothing to do — already complete.", flush=True)
        return

    cache = {}  # per-domain: prompt style differs between domains
    written = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for start in range(0, len(pending), args.item_chunk):
            if _SHUTDOWN_REQUESTED:
                print(f"[{domain}] stopping early (preemption). {written} written this attempt.",
                      flush=True)
                return

            chunk = pending[start:start + args.item_chunk]
            flat_texts, flat_meta = [], []
            for row in chunk:
                for oi, ot in enumerate(row.get(OPTIONS_KEY, [])):
                    if isinstance(ot, str) and ot.strip():
                        flat_texts.append(ot)
                        flat_meta.append((row, oi))
            if not flat_texts:
                continue

            paraphrased = paraphrase_texts(model, tokenizer, flat_texts, prompt_style, args, cache)
            for (row, oi), new_text in zip(flat_meta, paraphrased):
                row.setdefault("_new_options", {})[oi] = new_text

            for row in chunk:
                options = row.get(OPTIONS_KEY, [])
                if not options or not row.get(ANSWER_KEY):
                    continue
                m = row.get("_new_options", {})
                out_f.write(json.dumps({
                    "id": row["_item_id"],
                    "domain": domain,
                    QUESTION_KEY: row.get(QUESTION_KEY),
                    OPTIONS_KEY: [m.get(i, o) for i, o in enumerate(options)],
                    ANSWER_KEY: row.get(ANSWER_KEY),
                }, ensure_ascii=False) + "\n")
                written += 1
                row.pop("_new_options", None)

            out_f.flush()
            os.fsync(out_f.fileno())

            ci = start // args.item_chunk
            if ci % args.empty_cache_every_chunks == 0:
                torch.cuda.empty_cache()
            if ci % args.log_every_chunks == 0:
                total_done = len(completed) + written
                pct = 100.0 * total_done / max(1, len(rows))
                mem = torch.cuda.memory_allocated() / (1024 ** 3)
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[{domain}] progress {total_done}/{len(rows)} ({pct:.1f}%) "
                      f"| attempt: {written} | gpu={mem:.2f}GiB peak={peak:.2f}GiB "
                      f"| uniq_cached={len(cache)}", flush=True)

    print(f"[{domain}] DONE — {written} written this attempt "
          f"({len(completed) + written}/{len(rows)} total).", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "test"], required=True)
    p.add_argument("--sample_per_domain", type=int, default=1000)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--item_chunk", type=int, default=64,
                    help="Checkpoint granularity + packer window. Does NOT set peak memory.")
    p.add_argument("--batch_token_budget", type=int, default=24576,
                    help="THE memory control: batch_size * longest_seq must stay under this.")
    p.add_argument("--max_seqs_cap", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--max_input_len", type=int, default=512)
    p.add_argument("--dedup", dest="dedup", action="store_true", default=True)
    p.add_argument("--no_dedup", dest="dedup", action="store_false")
    p.add_argument("--log_every_chunks", type=int, default=5)
    p.add_argument("--empty_cache_every_chunks", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
    print(f"split={args.split} sample_per_domain={args.sample_per_domain} "
          f"token_budget={args.batch_token_budget} max_seqs_cap={args.max_seqs_cap} "
          f"item_chunk={args.item_chunk} dedup={args.dedup}", flush=True)
    print(f"Loading paraphraser: {args.model_name} (4-bit) ...", flush=True)
    model, tokenizer = load_model(args.model_name)
    print(f"Model loaded. padding_side={tokenizer.padding_side}", flush=True)

    for domain in ["math", "code", "stem"]:
        if _SHUTDOWN_REQUESTED:
            print("Exiting before next domain (preemption).", flush=True)
            break

        path = os.path.join(args.input_dir, f"{domain}.jsonl")
        if not os.path.exists(path):
            print(f"Skipping missing split: {path}", flush=True)
            continue

        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        for r in rows:
            r["_item_id"] = get_item_id(r)

        want_test = (args.split == "test")
        rows = [r for r in rows if is_test_id(r["_item_id"]) == want_test]

        rows.sort(key=lambda r: r["_item_id"])
        if len(rows) > args.sample_per_domain:
            rows = random.Random(args.seed).sample(rows, args.sample_per_domain)
            rows.sort(key=lambda r: r["_item_id"])

        process_domain(model, tokenizer, domain, rows,
                        os.path.join(args.output_dir, f"{domain}.jsonl"), args)

    if _SHUTDOWN_REQUESTED:
        print("Exiting code 1 so AML retries and resumes from checkpoint.", flush=True)
        sys.exit(1)
    print("All domains complete.", flush=True)


if __name__ == "__main__":
    main()
