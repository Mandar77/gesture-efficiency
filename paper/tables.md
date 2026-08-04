| Method | Source | Dataset | Params(M) | FLOPs(G) | Top-1(%) | bs8 FPS | bs1 FPS(indic.) | PeakVRAM(MB) | Disk(MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jester_compact3dcnn_16f172_30ep | ours | jester | 1.17 | 4.84 | 78.91 | 1589.7 | 748.6 | 686.3 | 4.482 |
| jester_compact3dcnn_8f224_30ep | ours | jester | 1.17 | 4.01 | 69.23 | 1858.0 | 735.8 | 592.6 | 4.482 |
| fp16_prune0.0 | ours | TODO | 3.11 | TODO | 93.48 | TODO | TODO | TODO | 6.061 |
| fp16_prune0.3 | ours | TODO | 3.11 | TODO | 9.93 | TODO | TODO | TODO | 6.061 |
| fp16_prune0.5 | ours | TODO | 3.11 | TODO | 3.56 | TODO | TODO | TODO | 6.061 |
| fp32_prune0.0 | ours | TODO | 3.11 | TODO | 93.49 | TODO | TODO | TODO | 12.031 |
| fp32_prune0.3 | ours | TODO | 3.11 | TODO | 9.93 | TODO | TODO | TODO | 12.031 |
| fp32_prune0.5 | ours | TODO | 3.11 | TODO | 3.56 | TODO | TODO | TODO | 12.031 |
| int8_ptq_prune0.0 | ours | TODO | 3.11 | TODO | 93.49 | TODO | TODO | TODO | 11.998 |
| int8_qat_prune0.0 | ours | TODO | 3.11 | TODO | TODO | TODO | TODO | TODO | 3.606 |
| jester_student_logit_feat_kd | ours | jester | 3.11 | 6.20 | 93.47 | 211.1 | 135.6 | 2463.8 | 12.031 |
| jester_student_logit_kd | ours | TODO | 3.11 | 6.20 | 93.46 | 211.1 | 135.6 | 2440.1 | 12.031 |
| jester_student_no_kd | ours | TODO | 3.11 | 6.20 | 93.26 | 211.1 | 135.6 | 2341.8 | 12.031 |
| briareo_rgb | ours | TODO | 1.20 | TODO | 65.97 | TODO | TODO | TODO | 4.600 |
| briareo_rgbd | ours | TODO | 2.47 | TODO | 68.75 | TODO | TODO | TODO | 9.450 |
| briareo_rgbdir | ours | TODO | 3.80 | TODO | 70.83 | TODO | TODO | TODO | 14.549 |
| jester_vitb_full_ft_lr5e5_8f224 | ours | jester | 100.00 | 270.09 | 85.41 | 64.6 | 56.1 | 2863.8 | 381.525 |
| jester_vitb_lora_r16_lr2e4_8f224 | ours | jester | 100.88 | 272.88 | 87.69 | 53.2 | 42.9 | 1638.8 | 384.932 |
| jester_vitb_lora_r16_sanity1ep | ours | jester | 100.88 | 272.88 | 78.12 | TODO | 27.2 | 1889.0 | 384.932 |
| jester_vits_adapter_8f224 | ours | jester | 26.42 | 71.78 | 86.50 | 158.4 | 44.2 | 593.3 | 100.890 |
| jester_vits_full_ft_8f224 | ours | jester | 25.23 | 68.06 | 83.20 | 188.0 | 68.3 | 747.0 | 96.297 |
| jester_vits_lora_8f224 | ours | jester | 25.45 | 68.76 | 86.49 | 147.3 | 45.5 | 720.5 | 97.172 |
| jester_vits_lora_normfix_sanity1ep | ours | jester | 25.45 | 68.76 | 78.88 | TODO | 9.7 | 754.8 | 97.172 |
| jester_vits_prompt_8f224 | ours | jester | 25.23 | 70.79 | 75.00 | 181.1 | 71.2 | 695.0 | 96.309 |
| smoke_compact3dcnn | ours | TODO | 0.00 | 0.02 | 6.25 | TODO | 2392.3 | 67.6 | 0.021 |
| MoViNet-A0 | reported (Kondratyuk et al., CVPR 2021) | Kinetics-600 | 3.10 | 2.71 | 71.50 | TODO | TODO | TODO | TODO |
| MoViNet-A1 | reported (Kondratyuk et al., CVPR 2021) | Kinetics-600 | 4.60 | 6.02 | 76.00 | TODO | TODO | TODO | TODO |
| ConvMixFormer | reported (Garg et al., WACV 2025) | NVGesture (RGB) | 13.57 | TODO | 76.04 | TODO | TODO | TODO | TODO |
| GestFormer | reported (Garg et al., CVPR 2024 WiCV) | NVGesture (5-modality) | 24.08 | TODO | 85.85 | TODO | TODO | TODO | TODO |
| DSTSA-GCN | reported (Cui et al., Neurocomputing 2025) | SHREC'17 (14G, skeleton) | 1.99 | 1.79 | 97.74 | TODO | TODO | TODO | TODO |

> Reported rows are published numbers on different datasets/protocols and are **not directly comparable** to our measured runs.
