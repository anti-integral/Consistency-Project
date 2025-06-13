#!/usr/bin/env bash
CKPT="ckpts/ctcm_ema_step400000.pt"   # update if you stopped earlier
python -m sampling.sample --config configs/cifar10_default.yaml --ckpt $CKPT --outdir samples
