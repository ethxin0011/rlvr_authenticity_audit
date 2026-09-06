"""
File: 10_rlvr_training.py  (v5 — memory-safe: 4-bit base model + gradient checkpointing)
Run as: AML job on rlvr2 (GPU) — run TWICE, once per --group.

Phase 2 causal intervention. Short GRPO-style RLVR training of Qwen3-1.7B (QLoRA-style: 4-bit
base + LoRA adapters) on 4-option MCQs built from either:
  --group treatment : original GooseReason data (authenticity artifact present)
  --group control   : paraphrase/restyle-matched data from 08 (artifact removed)

Reward is rule-based and fully verifiable: 1.0 if the emitted option letter matches the correct
letter, else 0.0. No LLM judge is involved, so no second verifier's bias enters the experiment.

CHANGE (2026-09-05) — OOM FIX, root cause confirmed from traceback:
  Previous version loaded the full 1.7B model in plain fp16 with NO quantization. On a 16GB T4,
  fp16 weights (~3.4GB) plus GRPO's num_generations=4 (4 sampled completions scored per prompt,
  each requiring full-model activations through all 28 layers for backprop, since gradients must
  flow through frozen base weights to reach the LoRA adapters) consumed 15.55 of 15.56 GiB at
  the very first training step — there was no memory margin at all, so this was never going to
  fit regardless of batch size tuning alone.

  Fix (matches the working QLoRA-style pattern already used in 08_construct_control_set.py):
    1. Base model loaded 4-bit (NF4) via BitsAndBytesConfig, same as the paraphraser — cuts
       weight memory from ~3.4GB to under ~1GB.
    2. prepare_model_for_kbit_training() before get_peft_model() — required for 4-bit + LoRA
       (casts norm layers to fp32, enables gradient flow into adapters).
    3. gradient_checkpointing=True — trades ~20-30% slower training for large activation-memory
       savings (recomputes activations during backward instead of storing all of them).
    4. num_generations 4 -> 2, per_device_train_batch_size 2 -> 1, gradient_accumulation_steps
       4 -> 8 (keeps the same effective batch size of 8, just spread differently, at far lower
       peak memory since fewer sequences are held in memory simultaneously).
    5. max_prompt_length 512 -> 384 (MCQ prompts here are well under this; the cap only matters
       for pathological outliers and reducing it trims some activation memory for free).
    6. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set before torch import, per the
       fragmentation hint the OOM error itself gave.

  Sample size / train-set-size parity from v4 is UNCHANGED and unaffected by this fix
  (--sample_per_domain still caps both groups identically).

Trains ONLY on the train partition (via common_split.is_test_id), so 12's evaluation set is
never seen during training.
"""

import os

# MUST precede torch's CUDA init — combats the fragmentation named in the OOM message.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import argparse
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

from common_split import get_item_id, is_test_id

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


def build_mcq_examples(rows, seed=42, max_options=4):
    rng = random.Random(seed)
    examples = []
    for row in rows:
        gold, distractors = get_gold_and_distractors(row)
        distractors = [d for d in distractors if isinstance(d, str) and d.strip()]
        if not isinstance(gold, str) or not gold.strip() or not distractors:
            continue
        options = ([gold] + distractors)[:max_options]
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        correct_letter = LETTERS[order.index(0)]  # index 0 was gold pre-shuffle
        block = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(shuffled))
        examples.append({
            "prompt": ("Read the options below and identify which one is correct. "
                       "Respond with ONLY the letter of the correct option.\n\n"
                       f"{block}\n\nAnswer:"),
            "correct_letter": correct_letter,
            "item_id": get_item_id(row),
        })
    return examples


def load_domain_rows(data_path, train_only=True, sample_per_domain=None, seed=42):
    """Loads rows, filters to the train partition, then applies an IDENTICAL per-domain
    sampling cap regardless of which asset data_path points at (raw or control-set) — this is
    what equalises training-set size between treatment and control (fixed in v4)."""
    domains = ["math", "code", "stem"]
    all_rows = []
    for domain in domains:
        path = os.path.join(data_path, f"{domain}.jsonl")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["domain"] = domain
                rows.append(r)

        if train_only:
            rows = [r for r in rows if not is_test_id(get_item_id(r))]

        if sample_per_domain is not None and len(rows) > sample_per_domain:
            rows.sort(key=lambda r: get_item_id(r))
            rows = random.Random(seed).sample(rows, sample_per_domain)

        print(f"[load] domain={domain} rows_after_filter_and_cap={len(rows)}", flush=True)
        all_rows.extend(rows)

    return all_rows


def make_reward_fn():
    letter_re = re.compile(r"[A-Ha-h]")

    def reward_fn(prompts=None, completions=None, correct_letter=None, **kwargs):
        rewards = []
        for completion, gold in zip(completions, correct_letter):
            text = completion if isinstance(completion, str) else str(completion)
            m = letter_re.search(text.strip())
            rewards.append(1.0 if (m and m.group(0).upper() == gold) else 0.0)
        return rewards

    return reward_fn


def load_model_4bit(model_name):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto"
    )
    return model, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["treatment", "control"], required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--num_train_steps", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_generations", type=int, default=2,
                    help="Reduced from 4 -> 2 for memory. Each generation requires full-model "
                         "activations through the whole forward pass for backprop.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                    help="Raised 4 -> 8 to keep effective batch size (=8) unchanged even though "
                         "per_device_train_batch_size dropped 2 -> 1.")
    p.add_argument("--max_prompt_length", type=int, default=384)
    p.add_argument("--max_completion_length", type=int, default=8)
    p.add_argument("--sample_per_domain", type=int, default=1000,
                    help="Caps items per domain AFTER the train-partition filter. Must match "
                         "08_construct_control_set.py's --sample_per_domain so treatment and "
                         "control train on equal-sized pools.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"[{args.group}] loading data from {args.data_path} "
          f"(sample_per_domain={args.sample_per_domain}) ...", flush=True)
    rows = load_domain_rows(args.data_path, train_only=True,
                             sample_per_domain=args.sample_per_domain, seed=args.seed)
    examples = build_mcq_examples(rows, seed=args.seed)
    print(f"[{args.group}] built {len(examples)} MCQ training examples", flush=True)
    if not examples:
        raise RuntimeError(
            "0 training examples. Check that data_path holds jsonl with the "
            "{question, options, answer} schema and that the split filter isn't excluding all rows."
        )

    dataset = Dataset.from_list(examples)

    print(f"[{args.group}] loading {args.model_name} (4-bit NF4) ...", flush=True)
    model, tokenizer = load_model_4bit(args.model_name)

    # Required setup order for 4-bit + LoRA (QLoRA-style): prepare THEN wrap with LoRA.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    print(f"[{args.group}] effective batch size = "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps} "
          f"({args.per_device_train_batch_size} x {args.gradient_accumulation_steps})", flush=True)

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        max_steps=args.num_train_steps,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=max(50, args.num_train_steps // 4),
        report_to=[],
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(),
        processing_class=tokenizer,
    )

    print(f"[{args.group}] GRPO training for {args.num_train_steps} steps ...", flush=True)
    trainer.train()

    final = os.path.join(args.output_dir, f"final_lora_{args.group}")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"[{args.group}] saved LoRA -> {final}", flush=True)


if __name__ == "__main__":
    main()
