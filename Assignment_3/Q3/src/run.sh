#!/bin/sh

python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN_wo_NORM' --niter 35 --init_epoch 34 --sample 1
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN_wo_NORM' --niter 35 --init_epoch 34 --sample 1

python main_vae.py --cuda 1 --exp_name 'VAE_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample 1
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample 1