---
title: Cognitive Fatigue Detector
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
models:
  - tdnathmlenthusiast/cognitive-fatigue-detector
---

# Cognitive Fatigue Detector

Real-time cognitive fatigue detection from eye-tracking signals via two-stage
cross-modal transfer learning.

**Stage 1** — Pretrained on GazeBase (881 subjects, 12,334 recordings, 1000 Hz)  
**Stage 2** — Fine-tuned on SEED-VIG (12 subjects, LOSO cross-validation, AUC = 0.906)

See the [model repository](https://huggingface.co/tdnathmlenthusiast/cognitive-fatigue-detector)
for the full architecture, training details, and all evaluation plots.
