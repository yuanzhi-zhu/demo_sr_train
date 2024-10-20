#!/bin/bash

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo " "

###########################################################################################################

# run with single GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
            main_train_psnr.py --opt options/train_msrresnet_psnr.json

# # run with two GPUs
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
#             main_train_psnr.py --opt options/train_msrresnet_psnr.json

# # run with cpu
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
#             main_train_psnr.py --opt options/train_msrresnet_psnr.json