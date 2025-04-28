import torch
import torch.nn as nn
from typing import Tuple


class VEncoder(nn.Module):
    """Encoder for VAE."""

    def __init__(
            self,
            n_input_features: int,
            n_hidden_neurons: int,
            n_latent_features: int,
    ):
        """
        :param n_input_features: number of input features (28 x 28 = 784 for MNIST)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_latent_features: size of the latent vector
        """
        super().__init__()
        self.input_to_hidden = nn.Linear(n_input_features, n_hidden_neurons)
        self.head1 = nn.Linear(n_hidden_neurons, n_latent_features)
        self.head2 = nn.Linear(n_hidden_neurons, n_latent_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode data to gaussian distribution params."""
        z_mean = None
        z_logvar = None
        h = self.input_to_hidden(x)
        act = nn.functional.relu(h)
        z_mean = self.head1(act)
        z_logvar = self.head2(act)
        return z_mean, z_logvar


class VDecoder(nn.Module):
    """Decoder for VAE."""

    def __init__(
            self,
            n_latent_features: int,
            n_hidden_neurons: int,
            n_output_features: int,
    ):
        """
        :param n_latent_features: number of latent features (same as in Encoder)
        :param n_hidden_neurons: number of neurons in hidden FC layer
        :param n_output_features: size of the output vector (28 x 28 = 784 for MNIST)
        """
        super().__init__()

        self.latent_to_hidden = nn.Linear(n_latent_features, n_hidden_neurons)
        self.hidden_to_output = nn.Linear(n_hidden_neurons, n_output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image."""
        r = None
        h = self.latent_to_hidden(z)
        act = nn.functional.relu(h)
        r = self.hidden_to_output(act)
        r = nn.functional.sigmoid(r)
        return r


class VariationalAutoencoder(nn.Module):
    """Variational Auto Encoder model."""

    def __init__(
            self,
            n_data_features: int,
            n_encoder_hidden_features: int,
            n_decoder_hidden_features: int,
            n_latent_features: int,
    ):
        """
        :param n_data_features: number of input and output features (28 x 28 = 784 for MNIST)
        :param n_encoder_hidden_features: number of neurons in encoder's hidden layer
        :param n_decoder_hidden_features: number of neurons in decoder's hidden layer
        :param n_latent_features: number of latent features
        """
        super().__init__()

        self.encoder = VEncoder(
            n_input_features=n_data_features,
            n_hidden_neurons=n_encoder_hidden_features,
            n_latent_features=n_latent_features,
        )
        self.decoder = VDecoder(
            n_latent_features=n_latent_features,
            n_hidden_neurons=n_decoder_hidden_features,
            n_output_features=n_data_features,
        )

        self.input_shape = None
        self.mu = None
        self.logvar = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        z = None
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through encoder network.
        takes: tensor of shape [batch_size x [image-size]] (input images batch)
        returns: tensor of shape [batch_size x latent_feature_size] (latent vector)
        """
        z = None
        if self.input_shape is None:
            self.input_shape = x.shape[1:]
        x = x.view(x.shape[0], -1)
        self.mu, self.logvar = self.encoder.forward(x)
        z = self.reparameterize(self.mu, self.logvar)
        return z

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        """Function to perform forward pass through decoder network.
        takes: tensor of shape [batch_size x latent_feature_size] (latent vector)
        returns: tensor of shape [batch_size x [image-size]] (reconstructed images batch)
        """
        r = None
        r = self.decoder.forward(z)
        return r.view(-1, *self.input_shape)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between N(mu, var) and N(0, 1)."""
        kld = None
        kld = 0.5 * torch.sum(torch.exp(self.logvar) + self.mu.pow(2) - 1 - self.logvar, dim=1)
        return kld.mean()

    def forward(self, x):
        z = self.encoder_forward(x)
        r = self.decoder_forward(z)
        return r


class VariationalAutoencoderLoss:

    def __init__(self, model: VariationalAutoencoder):
        self.reconstructionBCE = nn.BCELoss(reduction='sum')
        self.model = model

    def compute_loss_(self, recon_x, x):
        total_loss = None
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        recon_x = recon_x.view(batch_size, -1)

        kl_diver = self.model.kl_divergence()
        reconstruction_loss = self.reconstructionBCE(recon_x, x) / batch_size
        total_loss = reconstruction_loss + kl_diver
        return total_loss
