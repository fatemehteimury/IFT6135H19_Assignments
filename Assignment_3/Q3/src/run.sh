#!/bin/sh

# Run vae and wgan without normalization of input
python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN_wo_NORM' --niter 35 
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN_wo_NORM' --niter 35 
python main_vae.py --cuda 1 --exp_name 'VAE_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 

python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN_wo_NORM' --niter 35 --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN_wo_NORM' --niter 35 --init_epoch 34 --sample
python main_vae.py --cuda 1 --exp_name 'VAE_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR_wo_NORM' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample

# Run vae and wgan with normalization of input
python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN' --niter 35 --normalize
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN' --niter 35 --normalize
python main_vae.py --cuda 1 --exp_name 'VAE_AR' --generator_type 'Assignment_recom' --niter 35 --normalize
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR' --generator_type 'Assignment_recom' --niter 35 --normalize

python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN' --niter 35 --normalize --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN' --niter 35 --normalize --init_epoch 34 --sample
python main_vae.py --cuda 1 --exp_name 'VAE_AR' --generator_type 'Assignment_recom' --normalize --niter 35 --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR' --generator_type 'Assignment_recom' --normalize --niter 35 --init_epoch 34 --sample
