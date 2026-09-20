# Final Report — Authenticity-Artifact Audit of GooseReason-0.7M

## Phase 1 — Characterization

**No meaningful authenticity artifact detected overall** (AUROC=0.562, near chance).

| Split | GBM AUROC | LogReg AUROC | N |
|---|---|---|---|
| overall | 0.562 | 0.551 | 315499 |
| math | 0.584 | 0.598 | 101926 |
| code | 0.416 | 0.503 | 111159 |
| stem | 0.583 | 0.569 | 102414 |

## Phase 2 — Causal Intervention

**No causal exploitation detected**: treatment (0.021) not above control (0.027).

### Overall

| Group | Original | Neutralized | Adversarial | Exploitation gap |
|---|---|---|---|---|
| treatment | 0.283 | 0.212 | 0.191 | 0.021 |
| control | 0.299 | 0.233 | 0.207 | 0.027 |

### By domain

| Group | Domain | Original | Neutralized | Adversarial | Exploitation gap |
|---|---|---|---|---|---|
| treatment | code | 0.300 | 0.220 | 0.187 | 0.033 |
| treatment | math | 0.153 | 0.103 | 0.137 | -0.033 |
| treatment | stem | 0.397 | 0.313 | 0.250 | 0.063 |
| control | code | 0.330 | 0.263 | 0.207 | 0.057 |
| control | math | 0.167 | 0.123 | 0.150 | -0.027 |
| control | stem | 0.400 | 0.313 | 0.263 | 0.050 |