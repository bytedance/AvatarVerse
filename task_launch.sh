#!/bin/bash

export HF_HOME=~/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

python3 main.py --config configs/diffusion/diffusion-3d-controlnet.py --render_test --export_mesh --i_print=100 --i_weights=500 --mixed_precision fp16 --seed 0

