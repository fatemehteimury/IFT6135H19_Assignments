import torch
import torch.nn as nn
import torch.nn.functional as F
############################################################
#
# define generators, discriminator, encoder, and VAE
#
############################################################

class Generator(nn.Module):
    def __init__(self, ngf, nz, normalize=True):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.normalize = normalize

        self.linear_stack = nn.Sequential(
            # input is Z, going into Linear
            nn.Linear(nz, ngf*8*4*4),
            nn.ELU(),
            nn.Dropout(p=0.1),
            )

        self.conv_stack = nn.Sequential(
            # state size. (ngf*8) x 4 x 4

            nn.Conv2d(ngf*8, ngf*4, kernel_size=3, padding=2),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # state size. (ngf*4) x 12 x 12

            nn.Conv2d(ngf*4, ngf*2, kernel_size=3, padding=2),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # state size. (ngf*2) x 28 x 28

            nn.Conv2d(ngf*2, ngf*1, kernel_size=3, padding=2),
            nn.BatchNorm2d(ngf * 1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            # state size. (ngf*1) x 30 x 30

            nn.Conv2d(ngf*1,     3, kernel_size=3, padding=2),
            # state size.       3 x 32 x 32
            )

    def forward(self, x):
        # x.size = bs x nz
        x = self.linear_stack(x)
        output = self.conv_stack(x.view(-1, self.ngf*8, 4, 4))
        output = torch.tanh(output) if self.normalize else torch.sigmoid(output)
        # state size. bs x 3 x 32 x 32
        return output


class Discriminator(nn.Module):
    """
    Inspired by: https://github.com/pytorch/examples/tree/master/dcgan
    """
    def __init__(self, ndf):
        super(Discriminator, self).__init__()

        self.ndf = ndf

        self.linear_stack = nn.Sequential(
            # input is Z, going into Linear
            nn.Linear(ndf*8*4*4, 1),
            # nn.Sigmoid(),
            )

        self.conv_stack = nn.Sequential(
            # input is 3 x 32 x 32
  
            nn.Conv2d(      3, ndf * 2, 4, 2, 1),
            nn.ELU(),
            # state size. (ndf*2) x 32 x 32
  
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.ELU(),
            # state size. (ndf*4) x 16 x 16
  
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.ELU(),
            # state size. (ndf*8) x 4 x 4
        )

    def forward(self, x):
        x = self.conv_stack(x)
        output = self.linear_stack(x.view(-1, self.ndf*8*4*4)) 
        # state size. bs
        return output.squeeze(1)


class Encoder(nn.Module):

    def __init__(self, ngf, nz):
        super(Encoder, self).__init__()

        self.ngf = ngf

        self.mu_linear = nn.Linear(ngf*8*4*4, nz)
        self.logvar_linear = nn.Linear(ngf*8*4*4, nz)

        self.conv_stack = nn.Sequential(
            # input is          3 x 32 x 32

            nn.Conv2d(    3, ngf*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 1),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.AvgPool2d(kernel_size=2),
            # state size. (ngf*1) x 16 x 16

            nn.Conv2d(ngf*1, ngf*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.AvgPool2d(kernel_size=2),
            # state size. (ngf*2) x 8 x 8

            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            nn.AvgPool2d(kernel_size=2),
            # state size. (ngf*4) x 4 x 4

            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ELU(),
            nn.Dropout2d(p=0.1),
            # state size. (ngf*8) x 4 x 4
            )

    def forward(self, x):
        x = self.conv_stack(x)
        mu = self.mu_linear(x.view(-1, self.ngf*8*4*4))
        logvar = self.logvar_linear(x.view(-1, self.ngf*8*4*4))
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        # state size. bs x nz
        return z, mu, logvar



class DCGenerator(nn.Module):
    """
    Inspired by: https://github.com/pytorch/examples/tree/master/dcgan
    """
    def __init__(self, ngf, nz, normalize=True):
        super(DCGenerator, self).__init__()
        self.normalize = normalize
        self.main = nn.Sequential(
            # input is Z, going into a convolution

            nn.ConvTranspose2d(      nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d( ngf * 2,       3, 4, 2, 1, bias=False),
            # state size. 3 x 32 x 32
        )

    def forward(self, x):
        # x.size = bs x nz
        output = self.main(x.view(x.shape[0],x.shape[1],1,1))
        output = torch.tanh(output) if self.normalize else torch.sigmoid(output)
        # state size. bs x 3 x 32 x 32
        return output


class DCDiscriminator(nn.Module):
    """
    Inspired by: https://github.com/pytorch/examples/tree/master/dcgan
    """
    def __init__(self, ndf):
        super(DCDiscriminator, self).__init__()

        self.main = nn.Sequential(
            # input is 3 x 32 x 32

            nn.Conv2d(      3, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8,       1, 4, 1, 0, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.main(x)
        # state size. bs
        return output.view(-1, 1).squeeze(1)


class DCEncoder(nn.Module):

    def __init__(self, ngf, nz):
        super(DCEncoder, self).__init__()

        self.ngf = ngf

        self.mu_linear = nn.Linear(ngf*8*4*4, nz)
        self.logvar_linear = nn.Linear(ngf*8*4*4, nz)

        self.conv_stack = nn.Sequential(
            # input is          3 x 32 x 32

            nn.Conv2d(    3, ngf*1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(),
            # state size. (ngf*1) x 16 x 16

            nn.Conv2d(ngf*1, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 8 x 8

            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 4 x 4

            nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            )

    def forward(self, x):
        x = self.conv_stack(x)
        mu = self.mu_linear(x.view(-1, self.ngf*8*4*4))
        logvar = self.logvar_linear(x.view(-1, self.ngf*8*4*4))
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        # state size. bs x nz
        return z, mu, logvar



class VAE(nn.Module):
    
    def __init__(self, ngf, nz, generator_type='DCGAN', normalize=True):
        super(VAE, self).__init__()

        self.encoder = DCEncoder(ngf, nz) if generator_type=='DCGAN' else Encoder(ngf, nz)
        self.decoder = DCGenerator(ngf, nz, normalize) if generator_type=='DCGAN' else Generator(ngf, nz, normalize)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        output = self.decoder(z)
        return output, mu, logvar

    def sample(self, z):
        return self.decoder(z)