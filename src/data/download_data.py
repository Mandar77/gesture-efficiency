"""Print acquisition instructions for each dataset (no credentials hardcoded).

Datasets require registration/agreement, so this tool does not download blobs
automatically. It prints the current, documented manual steps and the expected
on-disk layout that `prepare_<dataset>.py` consumes. See DATA_LICENSES.md.

Usage:
    python src/data/download_data.py --dataset jester
    python src/data/download_data.py --dataset nvgesture
    python src/data/download_data.py --dataset shrec
"""

from __future__ import annotations

import argparse
import textwrap

INSTRUCTIONS = {
    "jester": """
    ==================== Jester (20BN-Jester v1) ====================
    License: Qualcomm "Data License Agreement - Research Use" (2022-07-28),
    research-only, NO redistribution. (NOT CC BY-NC-ND — that is legacy
    TwentyBN-era misinformation.)

    1. Obtain the dataset from the canonical live source:
         - Qualcomm developer portal:
             https://www.qualcomm.com/developer/software/jester-dataset
           Requires a Qualcomm ID login + accepting the Research-Use agreement.
           22.8 GB TGZ split into parts; 148,092 clips; 27 classes.
         - The old 20bn.com / TwentyBN host is dead (503).
         - Third-party mirrors (Kaggle toxicmender/20bn-jester, HuggingFace
           subsets) likely violate the no-redistribution clause — avoid for a
           clean paper.
    2. You need:
         - The frame archives (JPGs), concatenated in order then extracted:
             cat 20bn-jester-v1-?? | tar zx     (Linux)
             # Windows: concat the parts, then `tar -xzf jester.tgz`
           giving numbered clip folders: <root>/20bn-jester-v1/<clip_id>/<f>.jpg
         - The label CSVs: jester-v1-train.csv, jester-v1-validation.csv,
           jester-v1-labels.csv (and optionally jester-v1-test.csv). If the
           frame download does not bundle the CSVs, they are the official v1
           annotations mirrored at
           github.com/udacity/CVND---Gesture-Recognition/20bn-jester-v1/annotations
           (verify class list + 118,562/14,787 split counts before use).
    3. Place everything under data/jester/ so the layout is:
         data/jester/
           20bn-jester-v1/<clip_id>/000001.jpg ...
           jester-v1-train.csv         # "<clip_id>;<label text>"
           jester-v1-validation.csv
           jester-v1-labels.csv        # one label per line (27 classes)
    4. Build the index + integrity report:
         python src/data/prepare_jester.py --root data/jester
    """,
    "nvgesture": """
    ==================== NVGesture (RGB-D + IR) ====================
    License: governed by NVIDIA's bundled Participant Agreement.txt (research
    use; NO open CC license). Cite Molchanov et al., CVPR 2016.
    Canonical stats: 1532 videos, 25 classes, 20 subjects, 1050/482 train/test.

    1. Download from the official NVIDIA research page (External Links ->
       Google Drive folder):
         research.nvidia.com/publication/2016-06_online-detection-and-
           classification-dynamic-hand-gestures-recurrent-3d
       The folder distributes nvGesture_v1.7z.001 - .031 (~30 GB, 31 parts).
       The Drive link is occasionally flaky; a low-seed academic-torrent mirror
       exists as a fallback. Extract the 7z parts (7-Zip / p7zip).
    2. You need the per-clip modality videos (color, depth, IR) and the
       provided train/test split list (nvgesture_train_correct*.lst /
       nvgesture_test_correct*.lst). Report the 1050/482 split explicitly.
    3. Place under data/nvgesture/ preserving the release layout, e.g.:
         data/nvgesture/
           Video_data/class_XX/subjectY_.../  sk_color.avi, sk_depth.avi, ...
           nvgesture_train_correct_cvpr2016_v2.lst
           nvgesture_test_correct_cvpr2016_v2.lst
    4. Build the index:
         python src/data/prepare_nvgesture.py --root data/nvgesture
    """,
    "shrec": """
    ==================== SHREC'17 Track / DHG-14/28 (skeleton) ====================
    License: research use — cite De Smedt et al., 3DOR 2017. BACKUP TRACK ONLY,
    framed around efficiency (skeleton SOTA is saturated ~97.7% on SHREC 14G).
    Canonical split: 1960 train / 840 test (2800 seq, 14/28 classes, 22 joints).

    IMPORTANT: the official IMT Lille host (www-rech.telecom-lille.fr/
    shrec2017-hand) is DEFUNCT for downloads (its own page says the buttons are
    broken). Use a community mirror and disclose it in the paper — do NOT cite
    the dead official URL as if it were live:
      - Skeleton-only pickle (de-facto community source): shrec_data.pckl /
        shrec2017_skel-data.pckl on the Mines-Paris cloud, referenced by the
        Devineau et al. FG 2018 repo (guillaumephd/deep_learning_hand_gesture_
        recognition). Carries x_train/x_test/y_train_14|28/y_test_14|28.
      - Full depth+skeleton: redistributed by DG-STA / e2eET-Skeleton-Based-HGR
        (produces SHREC2017_3d_dictTVS_l250_s2800.pckl). Cite the 3DOR 2017
        paper as the source; disclose redistribution. Verify the 1960/840 split
        and 14/28 labels before reporting numbers.

    RECOMMENDED skeleton backup if download reliability matters: FPHA
    (guiggh/hand_pose_action, live official GitHub) — SHREC'17 and DHG-14/28
    both depend on the same failing IMT Lille infrastructure.

    Place under data/shrec/ with the train_gestures.txt / test_gestures.txt
    split files, then:  python src/data/prepare_shrec.py --root data/shrec
    """,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=sorted(INSTRUCTIONS))
    args = ap.parse_args()
    print(textwrap.dedent(INSTRUCTIONS[args.dataset]))
    print("No files were downloaded automatically (registration/licensing). "
          "Follow the steps above, then run the matching prepare_*.py.")


if __name__ == "__main__":
    main()
