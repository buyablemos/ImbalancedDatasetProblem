import torch
import torch.nn as nn

# Parametry modelu
IMG_SIZE = 128
CHANNELS = 3
LATENT_DIM = 256
HIDDEN_DIM = 1024

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        input_dim = IMG_SIZE * IMG_SIZE * CHANNELS

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(HIDDEN_DIM, int(HIDDEN_DIM * 0.85)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(HIDDEN_DIM * 0.85), int(HIDDEN_DIM * 0.65)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(HIDDEN_DIM * 0.65), HIDDEN_DIM // 2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc_var = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(HIDDEN_DIM // 2, int(HIDDEN_DIM * 0.65)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(HIDDEN_DIM * 0.65), int(HIDDEN_DIM * 0.85)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(HIDDEN_DIM * 0.85), HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim),
            nn.Tanh()
        )


    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar