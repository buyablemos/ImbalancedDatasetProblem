import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

# Parametry modelu
IMG_SIZE = 128
CHANNELS = 3
LATENT_DIM = 64
HIDDEN_DIM = 512

class VAE(nn.Module):
    def __init__(self,device='cpu',result_dir="results", load_pretrained=True):
        super(VAE, self).__init__()
        input_dim = IMG_SIZE * IMG_SIZE * CHANNELS

        self.result_dir = result_dir
        self.device=device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(HIDDEN_DIM, int(HIDDEN_DIM * 0.7)),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(int(HIDDEN_DIM * 0.7), HIDDEN_DIM // 2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc_var = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        if load_pretrained:
            self.load_models()


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

    def generate_from_accurate_params(self, mu, logvar, num_samples=1, device='cpu'):
        """
        Generuje próbki z rozkładu N(mu, exp(logvar))

        Args:
            model: VAE model
            mu: tensor (LATENT_DIM,) - średnia
            logvar: tensor (LATENT_DIM,) - log-wariancja
            num_samples: int - liczba próbek do wygenerowania
            device: str - urządzenie (cpu/cuda)
        Returns:
            Tensor (num_samples, CHANNELS, IMG_SIZE, IMG_SIZE)
        """
        self.eval()
        with torch.no_grad():

            mu = mu.expand(num_samples, -1).to(device)          # kształt: [num_samples, latent_dim]
            logvar = logvar.expand(num_samples, -1).to(device)  # kształt: [num_samples, latent_dim]

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # (num_samples, LATENT_DIM)
            z = mu + eps * std

            samples = self.decode(z)    # (num_samples, CHANNELS, IMG_SIZE, IMG_SIZE)
        return samples

    def load_models(self):

        model_path = os.path.join(self.result_dir, "vae_best.pth")

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            print("[INFO] Loaded best VAE weights.")
        else:
            print("[INFO] No VAE checkpoint found.")


    # Trenowanie
    def train_vae(self, epoch, dataloader, loss_fn):
        self.train()
        train_loss = 0
        for data in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon ,mu,logvar = self.forward(data)
            loss = loss_fn(recon_x=recon, x=data,mu=mu,logvar=logvar)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            #scheduler.step(loss)


        train_loss /= len(dataloader.dataset)
        print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f}')
        return train_loss



    def validate(self, epoch,dataloader, loss_fn):
        self.eval()
        validate_loss = 0
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                recon ,mu,logvar = self.forward(data)
                validate_loss += loss_fn(recon_x=recon, x=data,mu=mu,logvar=logvar).item()

        validate_loss /= len(dataloader.dataset)
        print(f'Validate Epoch: {epoch} | Loss: {validate_loss:.4f}')
        return validate_loss

    # Wizualizacja rekonstrukcji
    def visualize_reconstruction(self,epoch, dataloader):
        self.eval()
        with torch.no_grad():
            sample = next(iter(dataloader)).to(self.device)

            recon ,mu,logvar = self.forward(sample)

            #recon = (recon + 1) / 2

            fig, axes = plt.subplots(2, 16, figsize=(32, 8))
            for i in range(16):
                axes[0, i].imshow((sample[i]).cpu().permute(1, 2, 0))
                axes[0, i].axis('off')
                axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0))
                axes[1, i].axis('off')
            plt.savefig(f'{self.result_dir}/recon_{epoch%100}.png')
            plt.close()

