import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam, SGD
import torchvision.utils as vutils
import argparse
import time
import collections
import os
from tqdm import tqdm
import sys
from network.models import Encoder, Generator, DCGenerator, Discriminator, VAE, DCDiscriminator
from dataloader.data_loader import get_data_loader
from custom_callbacks.Loss_plotter import LossPlotter
from custom_callbacks.Logger import Logger
from os.path import join
import math

##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='WGAN-GP for SVHN dataset')

parser.add_argument('--data', type=str, default='../SVHN_dataset',
                    help='location of the SVHN data. Default: ../SVHN_dataset')
parser.add_argument('--generator_type', type=str, default='DCGAN',
                    help='type of generator (DCGAN or Assignment_recom). Default: DCGAN')
parser.add_argument('--batchSize', type=int, default=64,
                    help='input batch size. Default: 64')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector, Default: 100')
parser.add_argument('--ngf', type=int, default=64,
                    help='minimum number of feature maps in generator, Default: 64')
parser.add_argument('--ndf', type=int, default=64,
                    help='minimum number of feature maps in discriminator, Default: 64')
parser.add_argument('--niter', type=int, default=25,
                    help='number of epochs to train for. Default: 25')
parser.add_argument('--init_epoch', type=int, default=0,
                    help='initial epoch number to re-start training. Default: 25')
parser.add_argument('--nDpG', type=int, default=5,
                    help='number of discriminator iterations per generator iterations. Default: 5')
parser.add_argument('--GP_lambda', type=float, default=10.0,
                    help='Gradient Penalty Lambda Hyperparameter. Default: 10.0')
parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; SGD, ADAM. Default: ADAM')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate. Default: 0.0002')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. Default: 0.5')
parser.add_argument('--cuda', type=int, default=-1,
                    help='GPU number (0,1,2,..). Default: -1 i.e. CPU')
parser.add_argument('--outf', default='../Experiments',
                    help='folder to output images and model checkpoints. Default: ../Experiments')
parser.add_argument('--exp_name', default='WGAN_GP',
                    help='Name (identifier) of Experiment. Default: WGAN_GP')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed. Default: 1111')
parser.add_argument('--sample', type=int, default=0,
                    help='If True(1) will generate samples from pre-trained model. Default:0')
parser.add_argument('--normalize', action='store_true', 
                    help='enables normalization of samples')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

############################################################################
#
# Define Log paths and create corresponding dirs
#
############################################################################

log_path = join(args.outf, args.exp_name)
os.makedirs(log_path, exist_ok=True)
os.makedirs(join(log_path, 'weights'), exist_ok=True)
os.makedirs(join(log_path, 'visualize'), exist_ok=True)

argsdict['log_dir'] = log_path
with open (join(log_path,'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key+'    '+str(argsdict[key])+'\n')
print("===> Experiment Parameters saved")

############################################################################
#
# setup GPU or CPU
#
############################################################################
if args.cuda<0:
    print("===> Using CPU")
    device = torch.device("cpu")
else:
    print("===> Using GPU device: {}".format(args.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda")

#############################################################################
#
# set random seed for reproducibility
#
#############################################################################

np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)

############################################################################
#
# get Generator, Discriminator, data_loaders, and optimizers
#
############################################################################

G = DCGenerator(args.ngf, args.nz) if args.generator_type=='DCGAN' else Generator(args.ngf, args.nz)
D = DCDiscriminator(args.ndf) if args.generator_type=='DCGAN' else Discriminator(args.ndf) 

print("===> Generator and Discriminator Defined")

if args.optimizer=='SGD':
    optimizer_G = SGD(G.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    optimizer_D = SGD(D.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
else:
    optimizer_G = Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

print("===> Optimizers Initialized")

if args.init_epoch > 0:
    print("===> Loading pre-trained weight {}".format(args.init_epoch))
    weight_path = 'weights/model-{:04d}.pt'.format(args.init_epoch)
    checkp = torch.load(join(log_path, weight_path))
    G.load_state_dict(checkp['generator_state_dict'])
    optimizer_G.load_state_dict(checkp['optimizer_G_state_dict'])
    D.load_state_dict(checkp['discriminator_state_dict'])
    optimizer_D.load_state_dict(checkp['optimizer_D_state_dict'])


G = G.to(device)
D = D.to(device)

trainloader, validloader, testloader = get_data_loader(args.data, args.batchSize, args.normalize)
print("===> Data Loaders Initialized")


############################################################################
#
# Initialize Logger and LossPlotter
#
############################################################################

my_metric = ["D_fake", "D_real", "GP"]

my_loss = ["G_loss", "D_loss"]

logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
LP = LossPlotter(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)

print("===> Logger and LossPlotter Initialized")


############################################################################
#
# define Gradient Penalty loss and checkpoint
#
############################################################################

def calc_gradient_penalty(real_d, fake_d):

    # Calculate interpolation
    alpha = torch.rand(args.batchSize,1,1,1)
    alpha = alpha.expand_as(real_d)
    alpha = alpha.to(device)

    interpolates = alpha * real_d + ((1 - alpha) * fake_d)
    interpolates = interpolates.to(device)
    interpolates.requires_grad = True

    # Calculate probability of interpolated examples
    prob_interpolates = D(interpolates)

    # Calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(prob_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    # # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # # so flatten to easily take norm per example in batch
    gradients = gradients.view(args.batchSize, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = args.GP_lambda * ((gradients_norm - 1) ** 2).mean()

    # gradient_penalty = args.GP_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def checkpoint(epoch):
    w_path = 'weights/model-{:04d}.pt'.format(epoch)
    torch.save({'epoch': epoch, 'generator_state_dict': G.state_dict(), 'optimizer_G_state_dict': optimizer_G.state_dict(), 'discriminator_state_dict': D.state_dict(), 'optimizer_D_state_dict': optimizer_D.state_dict()}, join(log_path, w_path))


############################################################################
#
# Define train and test loop 
#
###########################################################################

one = torch.FloatTensor([1])
mone = one * -1

one = one.to(device)
mone = mone.to(device)

def train(epoch, data_loader):

    G.train()
    D.train()

    metric = np.zeros(len(my_metric)+len(my_loss))

    for iteration, batch in enumerate(tqdm(data_loader)):

        ############################
        # (1) Update D network: 
        ###########################

        D.zero_grad()

        # train with real
        real_input = batch[0].to(device)

        real_data = D(real_input)
        errD_real = real_data.mean()
        errD_real.backward(mone)

        # train with fake
        noise = torch.randn(args.batchSize, args.nz, device=device)
        fake_input = G(noise)

        fake_data = D(fake_input.detach())
        errD_fake = fake_data.mean()
        errD_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(real_input.data, fake_input.data)
        gradient_penalty.backward()

        errD = errD_fake.item() - errD_real.item() + gradient_penalty.item()

        optimizer_D.step()

        ############################
        # (2) Update G network:
        ###########################

        errG = 0.0
        # update generator for every args.nDpG discriminator updates 
        if iteration%args.nDpG == 0:

            G.zero_grad()

            noise = torch.randn(args.batchSize, args.nz, device=device)
            fake_input = G(noise)

            fake_data = D(fake_input)

            errG_D = fake_data.mean()
            errG_D.backward(mone)

            errG = - errG_D.item()

            optimizer_G.step()

    
        metric += np.array([errG*args.nDpG, errD, errD_fake.item(), - errD_real.item(), gradient_penalty.item()])

    return metric/len(data_loader)

      
def test(epoch, data_loader):

    G.eval()
    D.eval()

    metric = np.zeros(len(my_metric)+len(my_loss))

    frq = int(len(data_loader)/10)

    save_path = join(log_path, 'visualize', str(epoch))
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():

        for iteration, batch in enumerate(tqdm(data_loader)):

            ############################
            # (1) Update D network: 
            ###########################

            # train with real
            real_input = batch[0].to(device)

            real_data = D(real_input)
            errD_real = real_data.mean()

            # train with fake
            noise = torch.randn(args.batchSize, args.nz, device=device)
            fake_input = G(noise)

            fake_data = D(fake_input.detach())
            errD_fake = fake_data.mean()

            errD = errD_fake.item() - errD_real.item()

            ############################
            # (2) Update G network:
            ###########################

            noise = torch.randn(args.batchSize, args.nz, device=device)
            fake_input = G(noise)

            fake_data = D(fake_input)

            errG_D = fake_data.mean()

            errG = - errG_D.item()

            metric += np.array([errG, errD, errD_fake.item(), - errD_real.item(), 0.0])
        
            if iteration%frq==0:
                vutils.save_image(real_input, '{}/{}_real_samples.png'.format(save_path,str(iteration)), normalize=args.normalize, scale_each=args.normalize)
                vutils.save_image(fake_input.detach(), '{}/{}_fake_samples.png'.format(save_path,str(iteration)), normalize=args.normalize, scale_each=args.normalize)

    return metric/len(data_loader)


if not args.sample:
    ############################################################################ 
    #
    # Train Network 
    #
    ###########################################################################

    for epch in range(args.init_epoch, args.niter):
    
        start = time.time()	

        print("\n\n")
        print("Epoch:{}".format(epch))

        train_metric = train(epch, trainloader)
        print("===> Training     Epoch {}: G_Loss - {:.4f}, D_Loss - {:.4f}, D_fake - {:.4f}, D_real - {:.4f}, GP - {:.4f}".format(epch, train_metric[0], train_metric[1], train_metric[2], train_metric[3], train_metric[4]))

        valid_metric = test(epch, validloader)
        print("===> Validation   Epoch {}: G_Loss - {:.4f}, D_Loss - {:.4f}, D_fake - {:.4f}, D_real - {:.4f}, GP - {:.4f}".format(epch, valid_metric[0], valid_metric[1], valid_metric[2], valid_metric[3], valid_metric[4]))

        logger.to_csv(np.concatenate((train_metric, valid_metric)), epch)
        print("===> Logged All Metrics")

        LP.plotter()
        print("===> Plotted All Metrics")

        checkpoint(epch)
        print("===> Checkpoint saved")

        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epch, end-start))

else:
    if args.init_epoch == 0:
        print('===> You need pre-trained model. Use init_epoch argument to provide epoch number')
    
    else:
        ind_sample_path = join(log_path, 'samples', 'samples')
        sample_path = join(log_path, 'Qualitative_samples')

        os.makedirs(sample_path, exist_ok=True)
        os.makedirs(ind_sample_path, exist_ok=True)

        with torch.no_grad(): 
            
            # Interpolation in z and x space
            # generate random batch_size samples
            z_1 = torch.randn(args.batchSize, args.nz, device=device)
            z_2 = torch.randn(args.batchSize, args.nz, device=device)

            x_1 = G(z_1).detach()
            x_2 = G(z_2).detach()

            x = x_2.clone() 
            z = x_2.clone()

            for i in range(1,11):
                alpha = i/10
           
                t = (alpha*x_1) + ((1-alpha)*x_2)
                x = torch.cat([x, t.clone()])

                t = (alpha*z_1) + ((1-alpha)*z_2)
                z_t = G(t).detach()
                z = torch.cat([z, z_t.clone()])

            vutils.save_image(x, '{}/x_interpolation_samples.png'.format(sample_path), normalize=args.normalize, scale_each=args.normalize, nrow=args.batchSize)
            vutils.save_image(z, '{}/z_interpolation_samples.png'.format(sample_path), normalize=args.normalize, scale_each=args.normalize, nrow=args.batchSize)

            print('===> Interpolation in Z and X space saved')

            # perturbation in z space

            w = torch.randn(args.batchSize, args.nz, device=device)

            y = G(w).detach()

            eps = 3.0*torch.ones(args.batchSize, device=device)

            for i in range(args.nz):

                w_p_e = w.clone()
                w_p_e[:,i] += eps

                w_m_e = w.clone()
                w_m_e[:,i] -= eps

                y_p_e = G(w_p_e).detach()
                y_m_e = G(w_m_e).detach()

                img = torch.cat([y_m_e.clone(), y.clone(), y_p_e.clone()])
                vutils.save_image(img, '{}/z_perturbation_samples_dim_{}.png'.format(sample_path, str(i)), normalize=args.normalize, scale_each=args.normalize, nrow=args.batchSize)

            print('===> Perturbations in Z space saved')

            # individual samples for fid scores

            sample_counts = 0

            for i in range(math.ceil(1000/args.batchSize)):

                w = torch.randn(args.batchSize, args.nz, device=device)
                y = G(w).detach()

                for j in range(args.batchSize):
                    vutils.save_image(y[j:j+1,:,:,:], '{}/samples_{}.png'.format(ind_sample_path, str(sample_counts)), normalize=args.normalize, scale_each=args.normalize, nrow=1, padding=0)
                    sample_counts += 1

            print('===> Samples for FID saved \n\n\n')


