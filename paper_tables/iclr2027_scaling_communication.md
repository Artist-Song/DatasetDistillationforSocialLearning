# ICLR 2027 Communication Accounting

| Agents | Method | Native communication object | Unique/shared images | Image deliveries | Logical delivered auxiliary payload | Rounds to reported result |
|---:|---|---|---:|---:|---:|---:|
| 5 | Ours | Synthetic image packets + sender-local logits | 1,000 | 4,000 | sender logits: 640,000 (0.61 MiB) | 1 |
| 5 | Heuristic | Random real image packets + hard labels | 1,000 | 4,000 | none: 0 | 1 |
| 5 | FAST | FAST-selected real image packets + hard labels | 1,000 | 4,000 | none: 0 | 1 |
| 5 | Full Real | Full real-data image packets + hard labels | 50,000 | 200,000 | none: 0 | 1 |
| 5 | DeSA-CIL* | Synthetic anchors + iterative owner logits | 1,000 | 4,000 | owner logits: 32,000,000 (30.52 MiB) | 100 |
| 5 | MASC-complete* | Central real CC pool + expert/student parameters | 1,000 | - | model parameters: 598,687,920 (570.95 MiB) | 1 |
| 5 | FedRE | Shared head + entangled representations + mixture metadata | 0 | 0 | parameters/representations: 103,744,000 (98.94 MiB) | 100 |
| 10 | Ours | Synthetic image packets + sender-local logits | 1,000 | 9,000 | sender logits: 720,000 (0.69 MiB) | 1 |
| 10 | Heuristic | Random real image packets + hard labels | 1,000 | 9,000 | none: 0 | 1 |
| 10 | FAST | FAST-selected real image packets + hard labels | 1,000 | 9,000 | none: 0 | 1 |
| 10 | DeSA-CIL* | Synthetic anchors + iterative owner logits | 1,000 | 9,000 | owner logits: 36,000,000 (34.33 MiB) | 100 |
| 10 | MASC-complete* | Central real CC pool + expert/student parameters | 1,000 | - | model parameters: 1,532,858,240 (1461.85 MiB) | 1 |
| 10 | FedRE | Shared head + entangled representations + mixture metadata | 0 | 0 | parameters/representations: 207,368,000 (197.76 MiB) | 100 |
| 20 | Ours | Synthetic image packets + sender-local logits | 1,000 | 19,000 | sender logits: 760,000 (0.72 MiB) | 1 |
| 20 | Heuristic | Random real image packets + hard labels | 1,000 | 19,000 | none: 0 | 1 |
| 20 | FAST | FAST-selected real image packets + hard labels | 1,000 | 19,000 | none: 0 | 1 |
| 20 | DeSA-CIL* | Synthetic anchors + iterative owner logits | 1,000 | 19,000 | owner logits: 38,000,000 (36.24 MiB) | 100 |
| 20 | MASC-complete* | Central real CC pool + expert/student parameters | 1,000 | - | model parameters: 4,407,646,080 (4203.46 MiB) | 1 |
| 20 | FedRE | Shared head + entangled representations + mixture metadata | 0 | 0 | parameters/representations: 414,616,000 (395.41 MiB) | 100 |

Image counts and logical delivered auxiliary tensor bytes are intentionally reported separately.
