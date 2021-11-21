import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
import utils


class BetaVAE(nn.Module):

    def __init__(self,feature_dim,latent_size=8,beta=1):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            nn.LazyLinear(latent_size*4),
            nn.ReLU()
        )
        self.fc_mu = nn.LazyLinear(latent_size)
        self.fc_var = nn.LazyLinear(latent_size)

        # decoder
        self.decoder = nn.Sequential(
            nn.LazyLinear(feature_dim),
            nn.Softmax()
        )
        self.fc_z = nn.Linear(latent_size, latent_size*2)

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

class DNN(nn.Module):
    def __init__(self, input_size, neruons_num, dropout_prob):
        super().__init__()

        self.codename = 'dnn'

        self.layers = nn.Sequential(
            nn.LazyLinear(neurons_num[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[0]),
            nn.Linear(neurons_num[0], neurons_num[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[1]),
            nn.Linear(neurons_num[1], neurons_num[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob[2]),
            nn.Linear(neurons_num[2], 1)
        )
    
    def forward(self, batch):
        return self.layers(batch)
