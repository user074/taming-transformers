#!/bin/sh

python main.py --max_epochs 12 --base configs/custom_vqgan.yaml -t True --gpus 0, 1, 2, 3