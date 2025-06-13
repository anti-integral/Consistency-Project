# Continuous‑Time Consistency Model for CIFAR‑10

This repository contains a compact but fully‑featured implementation of a Continuous‑Time
Consistency Model (CTCM) that can be trained from scratch on a single high‑end GPU.
It reproduces the key ideas of **Lu & Song (2025)** — adaptive loss weighting,
tangent regularisation, and JVP‑based consistency training — while keeping the
codebase minimal (~1 k lines).

## Quick start
```bash
# 1. install deps
pip install -r requirements.txt
# 2. train
bash scripts/train.sh
# 3. sample & compute FID (needs ~10 GB RAM for 50 k images)
bash scripts/sample.sh
