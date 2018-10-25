import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch import tensor
import numpy as np

from abs_models import utils as u
from abs_models import loss_functions
from scipy.stats import multivariate_normal
from abs_models.cvae import CVAE

from abs_models import loss_functions



def get_gaussian_samples(n_samples, nd, device, mus,
                         fraction_to_dismiss=0.1, sample_sigma=1):
    """ returns nd coords sampled from gaussian in shape (n_samples, nd)
    """
    if mus is None:
        mus = np.zeros(nd)

    sigmas = np.diag(np.ones(nd)) * sample_sigma
    g = multivariate_normal(mus, sigmas)
    samples = g.rvs(size=int(n_samples / (1. - fraction_to_dismiss)))
    probs = g.pdf(samples)
    thresh = np.sort(probs)[-n_samples]
    samples = samples[probs >= thresh]

    samples = torch.from_numpy(samples[:, :, None, None].astype(np.float32)).to(device)
    return samples


def gd_inference_cvae(latent, x_inp, CVAE, device, n_classes=10, clip=5, lr=0.01, n_iter=20,
                   beta=1, dist_fct=loss_functions.squared_L2_loss):
    print('x_inp.shape', x_inp.shape)
    bs, n_ch, nx, ny = x_inp.shape
    with torch.enable_grad():
        latent = tensor(latent.data.clone().to(device), requires_grad=True)
        opti = optim.Adam([latent], lr=lr)
        for i in range(n_iter):
            ELBOs = []
            all_recs = []
            CVAE.eval()
            for j in range(n_classes):
                if i == n_iter - 1:
                    latent = latent.detach()  # no gradients in last run
                
                tensor_j = torch.from_numpy(np.repeat(j, len(latent[:, j]))).type(torch.LongTensor).to(device)
                rec = CVAE.decode(latent[:, j].squeeze(), tensor_j)#.cpu().data.numpy()

                ELBOs.append(loss_functions.ELBOs(rec,              # (bs, n_ch, nx, ny) 
                                                  latent[:, j],   # (bs, n_latent, 1, 1)
                                                  x_inp,            # (bs, n_ch, nx, ny)
                                                  beta=beta,        # should all be bs*num samples keeping
                                                  dist_fct=dist_fct,
                                                  auto_batch_size=None))
                if i == n_iter - 1:
                    all_recs.append(rec.view(bs, 1, n_ch, nx, ny).detach())

            ELBOs = torch.cat(ELBOs, dim=1)
            if i < n_iter - 1:
                loss = (torch.sum(ELBOs)) - 8./784./2  # historic reasons
                # backward
                opti.zero_grad()
                loss.backward()
                opti.step()
                latent.data = u.clip_to_sphere(latent.data, clip, channel_dim=2)
            else:
                opti.zero_grad()
                all_recs = torch.cat(all_recs, dim=1)
    return ELBOs.detach(), latent.detach(), all_recs


class CVAE_ABS(nn.Module):
    """
    n_samples: Number of samples to initially test 
    n_samples_grad: Do grad descent on top n_samples_grad of n_samples_try
    n_iter: number fo gradient descent steps
    beta: KLD weight
    sampler(size, ....????): function to return samples
    """
    def __init__(self, CVAE, n_samples, n_samples_grad, n_iter, beta, device,
                 n_labels=10, n_ch=1, nx=28, ny=28, 
                 sampler=get_gaussian_samples,
                 fraction_to_dismiss=0.1, clip=5, lr=0.05):
        super(CVAE_ABS, self).__init__()
        self.CVAE = CVAE.eval()
        # self.add_module(f'CVAE', CVAE)
        # for p in self.model.parameters():
        #     p.requires_grad=False

        self.n_samples = n_samples
        self.n_samples_grad = n_samples_grad
        self.n_iter = n_iter
        self.beta = beta
        self.device = device
        self.sampler = sampler

        self.n_latent = self.CVAE.latent_size
        self.n_labels = n_labels
        self.n_ch = n_ch
        self.nx = nx
        self.ny = ny

        self.fraction_to_dismiss = fraction_to_dismiss
        self.clip = clip
        self.lr = lr
        self.logit_scale = 440
        self.confidence_level = 0.000039
        self.name_check = 'MNIST_MSE'

    def rescale(self, logits):
        return logits

    def forward(self, x, return_more=False):
        sample_distance_function = loss_functions.squared_L2_loss
        sgd_distance_function=loss_functions.squared_L2_loss

        torch.cuda.manual_seed_all(101)
        torch.manual_seed(101)
        np.random.seed(101)
        bs = x.shape[0]

        # if n_iter == 0:
            # Do something weird to skip grad descent???

        # do the initial sampling and keep the top n_samples_grad:
        latent_samples = self.sampler(self.n_samples, self.n_latent, device=self.device, mus=None, fraction_to_dismiss=0.1, sample_sigma=1)

        # Generate an image from each sample for each class:
        gen_imgs = np.empty((self.n_labels, self.n_samples, self.n_ch, self.nx, self.ny))
        self.CVAE.eval()

        for label in range(self.n_labels):
            tensor_label = torch.from_numpy(np.repeat(label, self.n_samples)).type(torch.LongTensor).to(self.device)

            gen_imgs[label, ...] = self.CVAE.decode(latent_samples.squeeze(), tensor_label).cpu().data.numpy()
        gen_imgs = tensor(gen_imgs).type(torch.FloatTensor).to(self.device)
        print('done creating samples')

        # calculate the likelihood for all samples
        with torch.no_grad():
            all_ELBOs = loss_functions.ELBOs(
                            gen_imgs.view(1, self.n_labels, self.n_samples, self.n_ch, self.nx, self.ny),
                            latent_samples.view(1, 1, self.n_samples, self.n_latent, 1, 1),
                            x.view(bs, 1, 1, self.n_ch, self.nx, self.ny),
                            beta=self.beta, dist_fct=sample_distance_function,
                            auto_batch_size=8)
        x = x.view(bs, self.n_ch, self.nx, self.ny)

        # Keep only the top samples
        min_val_labels, min_val_labels_idx = torch.topk(all_ELBOs, k=self.n_samples_grad, dim=2, largest=False, sorted=True)
        # min_val_c, min_val_c_args = torch.min(all_ELBOs, dim=2)

        indices = min_val_labels_idx.view(bs * self.n_labels * self.n_samples_grad)

        # just throw the extra samples into the batch size dimension since it shouldnt matter. But where do the 1, 1 go?
        latent_samples_best = latent_samples[indices].view(bs*self.n_samples_grad, self.n_labels, self.n_latent, 1, 1)
        # l_v_best shape: (bs, n_classes, 8, 1, 1)
        # l_v_best = GM.l_v[n_samples][indices].view(bs, n_classes, n_latent, 1, 1)

        # Do gradient descent on the n_samples_grad*bs in latent_samples_best. 
        # This computes gradients in batches, through the auto batch(max bs=500)
        ELBOs, l_v_classes, reconsts = u.auto_batch(500, gd_inference_cvae, [latent_samples_best, x], self.CVAE, self.device,
                                                 n_classes=self.n_labels, clip=self.clip, lr=self.lr,
                                                 n_iter=self.n_iter, beta=self.beta, dist_fct=sgd_distance_function)


        ELBOs = self.rescale(ELBOs)  # ????????? class specific fine-scaling

        if return_more:
            p_c = u.confidence_softmax(-ELBOs * self.logit_scale, const=self.confidence_level,
                                       dim=1)
            return p_c, ELBOs, l_v_classes, reconsts
        else:
            return -ELBOs[:, :, 0, 0]   # like logits
