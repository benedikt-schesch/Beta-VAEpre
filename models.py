import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.LazyLinear(feature_dim)
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
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size
    
    def traversal(self, x, dimension, range, steps):
        results = []
        for student in x:
            res = []
            mu, logvar = self.encode(student)
            std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
            z = mu
            values = torch.linspace(-range, range, steps)
            for i in values:
                z[dimension] = mu[dimension]+i*std[dimension]
                rx = self.decode(z)
                res.append(rx)
            results.append(res)
        return results

class DNN(nn.Module):
    def __init__(self, neurons_num):
        super().__init__()
        modules = []
        for num in neurons_num:
            modules.append(nn.LazyLinear(num))
            modules.append(nn.LeakyReLU())
        modules.append(nn.LazyLinear(2))
        self.layers = nn.Sequential(*modules)
    
    def forward(self, batch):
        return self.layers(batch)
