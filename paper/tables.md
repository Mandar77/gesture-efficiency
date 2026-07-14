| Method | Source | Dataset | Params(M) | FLOPs(G) | Top-1(%) | FPS | Latency(ms) | PeakVRAM(MB) | Disk(MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jester_compact3dcnn_16f172_30ep | ours | jester | 1.17 | 4.84 | 78.91 | 459.7 | 2.18 | 686.3 | 4.482 |
| jester_student_logit_feat_kd | ours | jester | 3.11 | 6.20 | 93.47 | 110.4 | 9.06 | 2463.8 | 12.031 |
| jester_vits_adapter_8f224 | ours | jester | 26.42 | 71.78 | 86.50 | 57.6 | 17.35 | 593.3 | 100.890 |
| jester_vits_lora_8f224 | ours | jester | 25.45 | 68.76 | 86.49 | 56.1 | 17.83 | 720.5 | 97.172 |
| smoke_compact3dcnn | ours | TODO | 0.00 | 0.02 | 6.25 | 2392.3 | 0.42 | 67.6 | 0.021 |
| MoViNet-A0 | reported (Kondratyuk et al., CVPR 2021) | Kinetics-600 | 3.10 | 2.71 | 71.50 | TODO | TODO | TODO | TODO |
| MoViNet-A1 | reported (Kondratyuk et al., CVPR 2021) | Kinetics-600 | 4.60 | 6.02 | 76.00 | TODO | TODO | TODO | TODO |
| ConvMixFormer | reported (Garg et al., WACV 2025) | NVGesture (RGB) | 13.57 | TODO | 76.04 | TODO | TODO | TODO | TODO |
| GestFormer | reported (Garg et al., CVPR 2024 WiCV) | NVGesture (5-modality) | 24.08 | TODO | 85.85 | TODO | TODO | TODO | TODO |
| DSTSA-GCN | reported (Cui et al., Neurocomputing 2025) | SHREC'17 (14G, skeleton) | 1.99 | 1.79 | 97.74 | TODO | TODO | TODO | TODO |

> Reported rows are published numbers on different datasets/protocols and are **not directly comparable** to our measured runs.
