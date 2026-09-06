"""
File: 12_evaluate_models.py
Run as: AML job on rlvr2 (GPU) — one run evaluates BOTH policies.

Evaluates treatment- and control-trained LoRA policies on three held-out variants, all drawn
ONLY from the test partition:
  1. original     : raw GooseReason MCQs (artifact present)
  2. neutralized  : paraphrase/restyle-matched MCQs (artifact removed)
  3. adversarial  : neutralized gold + ONE original ("real-looking") distractor swapped in

Key metric — artifact exploitation gap = acc(neutralized) - acc(adversarial), reported overall
AND per domain. A large gap for treatment vs a small one for control is causal evidence that
training on artifact-bearing data induces shortcut reliance.

The per-item RNG seed uses stable_hash, so option order is reproducible across runs.

Output: eval_report.json
"""

import os
import json
import random
import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common_split import get_item_id, is_test_id, stable_hash

QUESTION_KEY = "question"
OPTIONS_KEY = "options"
ANSWER_KEY = "answer"
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


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
    return options[idx], [o for i, o in enumerate(options) if i != idx]


def load_domain_rows(data_path, filter_test=True):
    rows = []
    for domain in ["math", "code", "stem"]:
        path = os.path.join(data_path, f"{domain}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["domain"] = domain
                rows.append(r)
    if filter_test:
        rows = [r for r in rows if is_test_id(get_item_id(r))]
    return rows


def build_prompt(options, rng):
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    correct_letter = LETTERS[order.index(0)]  # options[0] must be gold
    block = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(shuffled))
    return ("Read the options below and identify which one is correct. "
            "Respond with ONLY the letter of the correct option.\n\n"
            f"{block}\n\nAnswer:"), correct_letter


def build_test_sets(raw_rows, neutral_rows, max_options=4):
    neutral_by_id = {get_item_id(r): r for r in neutral_rows}
    original, neutralized, adversarial = [], [], []

    for raw in raw_rows:
        item_id = get_item_id(raw)
        domain = raw.get("domain", "unknown")
        neu = neutral_by_id.get(item_id)
        if neu is None:
            continue  # paired comparison only

        rg, rd = get_gold_and_distractors(raw)
        rd = [d for d in rd if isinstance(d, str) and d.strip()]
        ng, nd = get_gold_and_distractors(neu)
        nd = [d for d in nd if isinstance(d, str) and d.strip()]
        if not rg or not ng or not rd or not nd:
            continue

        rng = random.Random(stable_hash(item_id) % (2 ** 31))

        p, c = build_prompt([rg] + rd[:max_options - 1], rng)
        original.append({"prompt": p, "correct_letter": c, "item_id": item_id, "domain": domain})

        p, c = build_prompt([ng] + nd[:max_options - 1], rng)
        neutralized.append({"prompt": p, "correct_letter": c, "item_id": item_id, "domain": domain})

        p, c = build_prompt([ng, rd[0]] + nd[:max_options - 2], rng)
        adversarial.append({"prompt": p, "correct_letter": c, "item_id": item_id, "domain": domain})

    return {"original": original, "neutralized": neutralized, "adversarial": adversarial}


@torch.no_grad()
def run_eval(model, tokenizer, examples, batch_size=8, max_new_tokens=8):
    """Returns (overall_accuracy, {domain: accuracy})."""
    letter_re = re.compile(r"[A-Ha-h]")
    correct_total = 0
    dom_correct, dom_total = {}, {}

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True
        ) for ex in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=768).to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        for ex, text in zip(batch, decoded):
            m = letter_re.search(text.strip())
            pred = m.group(0).upper() if m else None
            d = ex["domain"]
            dom_total[d] = dom_total.get(d, 0) + 1
            if pred == ex["correct_letter"]:
                correct_total += 1
                dom_correct[d] = dom_correct.get(d, 0) + 1

    overall = correct_total / len(examples) if examples else float("nan")
    per_domain = {d: dom_correct.get(d, 0) / dom_total[d] for d in dom_total}
    return overall, per_domain


def load_policy(base_model_name, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                 torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_test_path", type=str, required=True)
    p.add_argument("--neutral_test_path", type=str, required=True)
    p.add_argument("--treatment_lora_path", type=str, required=True)
    p.add_argument("--control_lora_path", type=str, required=True)
    p.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_rows = load_domain_rows(args.raw_test_path, filter_test=True)
    neutral_rows = load_domain_rows(args.neutral_test_path, filter_test=False)  # already test-only
    test_sets = build_test_sets(raw_rows, neutral_rows)
    print({k: len(v) for k, v in test_sets.items()}, flush=True)

    if any(len(v) == 0 for v in test_sets.values()):
        raise RuntimeError(
            "A test set is empty — raw_test_path and neutral_test_path must share item_ids. "
            "Both derive ids from the original question text via common_split.get_item_id."
        )

    report = {}
    for group, lora in [("treatment", args.treatment_lora_path),
                        ("control", args.control_lora_path)]:
        print(f"Loading {group} policy from {lora} ...", flush=True)
        model, tokenizer = load_policy(args.base_model_name, lora)

        overall_accs, per_domain_accs = {}, {}
        for split_name, examples in test_sets.items():
            o, pd_ = run_eval(model, tokenizer, examples)
            overall_accs[split_name] = o
            per_domain_accs[split_name] = pd_
            print(f"  [{group}] {split_name}: overall={o:.4f} by-domain={pd_}", flush=True)

        gap = overall_accs["neutralized"] - overall_accs["adversarial"]
        gap_by_domain = {
            d: per_domain_accs["neutralized"].get(d, float("nan"))
               - per_domain_accs["adversarial"].get(d, float("nan"))
            for d in set(per_domain_accs["neutralized"]) | set(per_domain_accs["adversarial"])
        }
        report[group] = {
            "overall": overall_accs,
            "per_domain": per_domain_accs,
            "artifact_exploitation_gap": gap,
            "artifact_exploitation_gap_by_domain": gap_by_domain,
        }

        del model
        torch.cuda.empty_cache()

    out = os.path.join(args.output_dir, "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved -> {out}", flush=True)

    print("\n=== Artifact exploitation gap (overall) ===", flush=True)
    for g in ["treatment", "control"]:
        print(f"  {g}: {report[g]['artifact_exploitation_gap']:.4f}", flush=True)
    print("\n=== By domain ===", flush=True)
    for g in ["treatment", "control"]:
        print(f"  {g}: {report[g]['artifact_exploitation_gap_by_domain']}", flush=True)


if __name__ == "__main__":
    main()
