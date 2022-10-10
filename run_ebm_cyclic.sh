#!/bin/bash
DATASET=$1
SRC=$2
TGT=$3
EXP=$4
export CUDA_VISIBLE_DEVICES=""
python adapt_cyclic.py --n_gpu 1 --dataset $DATASET --source $SRC --target $TGT \
	--channel_mul 8 --langevin_step 20 --langevin_lr 0.1 --lr 0.001  --beta1 0.5 --beta2 0.999 \
	--n_embed 512 --embed_dim 128 --batch_size 16 \
	--ae_ckpt vqvae_best_mm_whs_tfrec.pt \
	--data_root ../datasets/ \
	--expt $4
