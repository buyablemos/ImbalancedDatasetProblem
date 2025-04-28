import torch
import torch.nn as nn

# Parametry modelu
IMG_SIZE = 128
CHANNELS = 3
LATENT_DIM = 256
HIDDEN_DIM = 1024


class CNNVAE(nn.Module):
    def __init__(self):
        super(CNNVAE, self).__init__()

        # Encoder (konwolucje)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, IMG_SIZE/2, IMG_SIZE/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),         # [B, 64, IMG_SIZE/4, IMG_SIZE/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),        # [B, 128, IMG_SIZE/8, IMG_SIZE/8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),       # [B, 256, IMG_SIZE/16, IMG_SIZE/16]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),       # [B, 512, IMG_SIZE/32, IMG_SIZE/32]
            nn.ReLU()
        )

        # Ustalanie rozmiaru po przejściu przez encoder
        self.flatten_dim = 512 * (IMG_SIZE // 32) * (IMG_SIZE // 32)

        # Fully connected warstwy do latentnej przestrzeni
        self.fc_mu = nn.Linear(self.flatten_dim, LATENT_DIM)
        self.fc_var = nn.Linear(self.flatten_dim, LATENT_DIM)

        # Decoder (transponowane konwolucje)
        self.decoder_fc = nn.Sequential(
            nn.Linear(LATENT_DIM, self.flatten_dim),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, IMG_SIZE/16, IMG_SIZE/16]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # [B, 128, IMG_SIZE/8, IMG_SIZE/8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # [B, 64, IMG_SIZE/4, IMG_SIZE/4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),     # [B, 32, IMG_SIZE/2, IMG_SIZE/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1),  # [B, CHANNELS, IMG_SIZE, IMG_SIZE]
            nn.Sigmoid()  # Wartości wyjściowe w zakresie [0, 1]
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 512, IMG_SIZE // 32, IMG_SIZE // 32)  # Reshape do rozmiaru przed dekonwolucją
        return self.decoder_conv(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar