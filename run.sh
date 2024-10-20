#!/bin/bash

echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo " "

###########################################################################################################

# run PSNR with single GPU
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
            main_train_psnr.py --opt options/train_msrresnet_psnr.json

# # run PSNR with two GPUs
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
#             main_train_psnr.py --opt options/train_msrresnet_psnr.json

# # run PSNR with cpu
# CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=1 \
#             main_train_psnr.py --opt options/train_msrresnet_psnr.json

###########################################################################################################

# # run GAN with single GPU
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
#             main_train_gan.py --opt options/train_msrresnet_gan.json

# # run PSNR with two GPUs
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
#             main_train_gan.py --opt options/train_msrresnet_gan.json

# # run PSNR with cpu
# CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=1 \
#             main_train_gan.py --opt options/train_msrresnet_gan.json