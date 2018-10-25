import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class CVAE(nn.Module):
    """CVAE based off https://arxiv.org/pdf/1805.09190v3.pdf
    ??? SHould we use global avg pooling and a 1x1 conv to get mu, sigma? Or even no 1x1, just normal conv.

    should the first fc in deconv be making the output batch*8*7*7???
    """
    def __init__(self, latent_size=8, img_size=28, num_labels=10):
        super(CVAE, self).__init__()
        self.num_labels = num_labels
        self.latent_size = latent_size
        self.img_size = img_size
        self.linear_size = 7*7*16

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv4 = nn.Conv2d(64, 2*8, kernel_size=5, stride=1, padding=2, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_bn3 = nn.BatchNorm2d(16)

        # FC
        self.fc_mu = nn.Linear(self.linear_size+self.num_labels, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size+self.num_labels, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size+self.num_labels, self.linear_size)


    def forward(self, x, c, deterministic=False):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

    def encode(self, x, c):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, x, c):
        x = torch.cat((x, self.to_one_hot(c).type(x.type())), dim=1)
        x = self.fc_dec(x)
        x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, logvar, deterministic=False):
        if deterministic:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def loss(self, output, x, KLD_weight=1, info=False):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        recon_x, mu, logvar = output
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
        # loss = Variable(BCE+KLD_weight*KLD, requires_grad=True)
        # if info:
        #     return loss, BCE, KLD
        # return loss
        return BCE + KLD

    def to_one_hot(self, y):
        y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size()[0], self.num_labels).type(y.type())
        y_onehot.scatter_(1, y, 1)
        return y_onehot