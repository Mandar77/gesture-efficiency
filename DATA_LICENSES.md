# Dataset Licenses & Acquisition

This project **does not redistribute** any dataset. Each dataset must be
obtained from its official source under its own license. Record and respect the
terms below (BRIEF §2).

---

## Jester (20BN-Jester v1) — PRIMARY (RGB temporal)

- **Content:** 27 gesture classes, ~148,092 clips (≈118,562 train / 14,787 val
  / 14,743 test), frames as JPGs at 12 fps, ~23 GB.
- **License:** **Qualcomm "Data License Agreement – Research Use"** (dated
  2022-07-28) — **research-only, no redistribution.** This is the *current*
  license on Qualcomm's developer portal; it is **NOT** CC BY-NC-ND 4.0 (that
  is legacy TwentyBN-era misinformation still repeated by some third-party
  sources). Consequence for this repo: do **not** redistribute the frames or
  any processed derivative; release code + instructions only. Non-commercial /
  research use is permitted under the agreement accepted at download.
- **Citation:** Materzynska, Berger, Bax, Memisevic, *The Jester Dataset: A
  Large-Scale Video Dataset of Human Gestures*, ICCV Workshops 2019
  (arXiv:1909.05165).
- **Acquisition:** Originally 20BN/TwentyBN (`20bn.com` now dead, 503s); the
  canonical live source is the **Qualcomm developer portal**
  (`qualcomm.com/developer/software/jester-dataset`), which requires a Qualcomm
  ID login and acceptance of the Research-Use agreement. 22.8 GB TGZ split into
  ~1–10 GB parts; 148,092 clips; 27 classes (older sources saying "25" are
  outdated). Third-party mirrors (Kaggle `toxicmender/20bn-jester`, HuggingFace
  subsets) likely violate the no-redistribution clause — prefer the gated
  Qualcomm source. See `src/data/download_data.py --dataset jester`.
- **Provenance of this copy:** frame archives (`20bnjester-v1-00/01/02`, 21.36 GB
  concatenated, `cat parts | tar zx`) obtained by the user from the Qualcomm AI
  Research distribution. The annotation CSVs (`jester-v1-labels.csv`,
  `jester-v1-train.csv` = 118,562 rows, `jester-v1-validation.csv` = 14,787 rows)
  were **not** in that distribution and were fetched from the community mirror
  `github.com/udacity/CVND---Gesture-Recognition` (`20bn-jester-v1/annotations/`).
  The 27-class label list matches the official Qualcomm instructions PDF and the
  train/val counts match the official splits; clip-id alignment against the
  extracted frames was verified before use. These CSVs are annotation metadata
  for the same v1 release, used under the same Qualcomm Research-Use terms.
  _(Note: the frame archives were obtained by the user; provenance is disclosed
  transparently rather than claiming the dead official author host works.)_

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
