"""
File: 14_aggregate_report.py
Run in: notebook on cpurlvr (final step, after all jobs complete).

Combines Phase 1 (classifier_report.json) and Phase 2 (eval_report.json) into final_report.md
plus two charts, using the pre-registered three-outcome interpretation so the write-up is
mechanical rather than another place to introduce a hard gate.

Reports per-domain exploitation gaps alongside overall, so the math/stem vs code
construction-methodology difference is a stated finding rather than lost in an aggregate.

Usage:
    python 14_aggregate_report.py \
        --classifier_report_path ./results/classifier_report.json \
        --eval_report_path ./phase2_download/.../eval_report.json \
        --output_dir ./results_v2
"""

import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def interpret_phase1(classifier_reports):
    overall = next(r for r in classifier_reports if r["split"] == "overall")
    auc = overall["gbm_auroc"]
    if auc >= 0.75:
        return f"**Strong authenticity artifact detected** (AUROC={auc:.3f})."
    if auc >= 0.60:
        return f"**Moderate authenticity artifact detected** (AUROC={auc:.3f})."
    return (
        f"**No meaningful authenticity artifact detected overall** (AUROC={auc:.3f}, near "
        "chance). The domain breakdown below shows this varies by construction methodology: "
        "`code` uses minimal-mutation bug-injection distractors (near-identical to gold), while "
        "`math`/`stem` use freely LLM-generated distractors."
    )


def interpret_phase2(eval_report):
    t = eval_report["treatment"]["artifact_exploitation_gap"]
    c = eval_report["control"]["artifact_exploitation_gap"]
    diff = t - c
    if diff > 0.05:
        return (f"**Causal exploitation confirmed** overall: treatment gap ({t:.3f}) exceeds "
                f"control ({c:.3f}), diff={diff:.3f}.")
    if diff > 0.0:
        return (f"**Weak/partial exploitation signal** overall: treatment ({t:.3f}) vs control "
                f"({c:.3f}), diff={diff:.3f}.")
    return (f"**No causal exploitation detected** overall: treatment ({t:.3f}) is not above "
            f"control ({c:.3f}). A detectable artifact is not necessarily an exploited one — "
            "a valuable robustness result.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier_report_path", type=str, required=True)
    p.add_argument("--eval_report_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.classifier_report_path) as f:
        classifier_reports = json.load(f)
    with open(args.eval_report_path) as f:
        eval_report = json.load(f)

    lines = [
        "# Final Report — Authenticity-Artifact Audit of GooseReason-0.7M\n",
        "## Phase 1 — Characterization (Sub-claim A)\n",
        interpret_phase1(classifier_reports) + "\n",
        "| Split | GBM AUROC | LogReg AUROC | N |",
        "|---|---|---|---|",
    ]
    for r in classifier_reports:
        lines.append(f"| {r['split']} | {r['gbm_auroc']:.3f} | {r['logreg_auroc']:.3f} "
                     f"| {r['n_samples']} |")

    lines += [
        "\n## Phase 2 — Causal Intervention (Sub-claim B)\n",
        interpret_phase2(eval_report) + "\n",
        "### Overall\n",
        "| Group | Original | Neutralized | Adversarial | Exploitation gap |",
        "|---|---|---|---|---|",
    ]
    for g in ["treatment", "control"]:
        o = eval_report[g]["overall"]
        lines.append(f"| {g} | {o['original']:.3f} | {o['neutralized']:.3f} | "
                     f"{o['adversarial']:.3f} | "
                     f"{eval_report[g]['artifact_exploitation_gap']:.3f} |")

    lines += [
        "\n### By domain\n",
        "Key finding: `code` uses a different distractor-construction methodology than "
        "`math`/`stem` (minimal-mutation bug injection vs free LLM generation).\n",
        "| Group | Domain | Original | Neutralized | Adversarial | Exploitation gap |",
        "|---|---|---|---|---|---|",
    ]
    for g in ["treatment", "control"]:
        pd_ = eval_report[g]["per_domain"]
        gaps = eval_report[g]["artifact_exploitation_gap_by_domain"]
        domains = sorted(set(pd_.get("original", {})) | set(pd_.get("neutralized", {}))
                         | set(pd_.get("adversarial", {})))
        for d in domains:
            lines.append(
                f"| {g} | {d} | {pd_['original'].get(d, float('nan')):.3f} | "
                f"{pd_['neutralized'].get(d, float('nan')):.3f} | "
                f"{pd_['adversarial'].get(d, float('nan')):.3f} | "
                f"{gaps.get(d, float('nan')):.3f} |"
            )

    report_path = os.path.join(args.output_dir, "final_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved final report -> {report_path}")

    # Overall gap
    plt.figure(figsize=(5, 4))
    groups = ["treatment", "control"]
    plt.bar(groups, [eval_report[g]["artifact_exploitation_gap"] for g in groups],
            color=["#d9534f", "#5cb85c"])
    plt.ylabel("Artifact exploitation gap")
    plt.title("Overall: treatment vs. control")
    plt.savefig(os.path.join(args.output_dir, "exploitation_gap.png"),
                bbox_inches="tight", dpi=150)

    # Per-domain grouped bars
    domains = sorted(set(eval_report["treatment"]["artifact_exploitation_gap_by_domain"]))
    if domains:
        x = range(len(domains))
        w = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar([i - w / 2 for i in x],
                [eval_report["treatment"]["artifact_exploitation_gap_by_domain"].get(d, 0)
                 for d in domains], w, label="treatment", color="#d9534f")
        plt.bar([i + w / 2 for i in x],
                [eval_report["control"]["artifact_exploitation_gap_by_domain"].get(d, 0)
                 for d in domains], w, label="control", color="#5cb85c")
        plt.xticks(list(x), domains)
        plt.ylabel("Artifact exploitation gap")
        plt.title("By domain: treatment vs. control")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "exploitation_gap_by_domain.png"),
                    bbox_inches="tight", dpi=150)

    print("Saved plots -> exploitation_gap.png, exploitation_gap_by_domain.png")


if __name__ == "__main__":
    main()
