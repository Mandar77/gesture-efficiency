# Dataset Licenses & Acquisition

This project **does not redistribute** any dataset. Each dataset must be
obtained from its official source under its own license. Record and respect the
terms below (BRIEF §2).

---

## Jester (20BN-Jester v1) — PRIMARY (RGB temporal)

- **Content:** 27 gesture classes, ~148,092 clips (≈118,562 train / 14,787 val
  / 14,743 test), frames as JPGs at 12 fps, ~23 GB.
- **License:** **CC BY-NC-ND 4.0** — Attribution, **Non-Commercial**,
  **No-Derivatives**. Fine for academic research and reporting results; note
  the no-derivatives clause when releasing any derived artifact (do **not**
  redistribute processed frames; release code + instructions instead).
- **Citation:** Materzynska et al., *The Jester Dataset: A Large-Scale Video
  Dataset of Human Gestures*, ICCV Workshops 2019.
- **Acquisition:** Originally hosted by 20BN/Qualcomm; now commonly mirrored on
  Kaggle and academic mirrors. See `src/data/download_data.py --dataset jester`
  for current instructions. Registration may be required.

## NVGesture — PRIMARY (genuine multimodal RGB-D + IR)

- **Content:** 25 dynamic gesture classes, 1,532 clips (1,050 train / 482
  test); modalities **RGB, depth, IR** (optical flow derivable). This is the
  *real* multimodal track that replaces the dropped simulated depth/EMG.
- **License:** Released for **research use** by NVIDIA. Cite the paper and
  follow the terms on the official page.
- **Citation:** Molchanov et al., *Online Detection and Classification of
  Dynamic Hand Gestures with Recurrent 3D CNNs*, CVPR 2016.
- **Acquisition:** Official NVIDIA project page (registration/agreement).
  See `src/data/download_data.py --dataset nvgesture`.

## SHREC'17 Track + DHG-14/28 — BACKUP (skeleton, reuses MediaPipe)

- **Content:** 3D hand-skeleton sequences (22 joints, + depth), 14/28-class
  protocols, ~2,800 sequences each. Used only if the foundation-model track
  won't fit, or as an efficiency-only side experiment. Skeleton SOTA is
  saturated (~97.7% on SHREC 14G), so any skeleton work here is framed around
  **efficiency**, never accuracy.
- **License:** Research use; cite the SHREC'17 track / DHG papers.
- **Acquisition:** Official SHREC'17 gesture track page.

---

**No credentials are hardcoded anywhere.** `download_data.py` prints manual
steps when registration is required rather than embedding secrets.
