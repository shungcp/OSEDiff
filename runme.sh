#!/bin/bash
time python testv2.py -i preset/datasets/test_dataset/input -o preset/datasets/test_dataset/output --osediff_path preset/models/osediff.pkl --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base --ram_ft_path pretrain/DAPE.pth --ram_path pretrain/ram_swin_large_14m.pth
