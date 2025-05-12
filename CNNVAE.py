import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Subset

# Parametry modelu
IMG_SIZE = 128
CHANNELS = 3
LATENT_DIM = 256


class CNNVAE(nn.Module):
    def __init__(self, device='cpu', result_dir="results", load_pretrained=True):
        super(CNNVAE, self).__init__()

        self.result_dir = result_dir
        self.device = device

        # Encoder (konwolucje)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, IMG_SIZE/2, IMG_SIZE/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, IMG_SIZE/4, IMG_SIZE/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, IMG_SIZE/8, IMG_SIZE/8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, IMG_SIZE/16, IMG_SIZE/16]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, IMG_SIZE/32, IMG_SIZE/32]
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
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, IMG_SIZE/8, IMG_SIZE/8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, IMG_SIZE/4, IMG_SIZE/4]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, IMG_SIZE/2, IMG_SIZE/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, CHANNELS, kernel_size=4, stride=2, padding=1),  # [B, CHANNELS, IMG_SIZE, IMG_SIZE]
            nn.Sigmoid()  # Wartości wyjściowe w zakresie [0, 1]
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        if load_pretrained:
            self.load_models()

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

    def generate_from_accurate_params(self, mu, logvar, num_samples=1):

        self.eval()
        with torch.no_grad():
            mu = mu.expand(num_samples, -1).to(self.device)  # kształt: [num_samples, latent_dim]
            logvar = logvar.expand(num_samples, -1).to(self.device)  # kształt: [num_samples, latent_dim]

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # (num_samples, LATENT_DIM)
            z = mu + eps * std

            samples = self.decode(z)  # (num_samples, CHANNELS, IMG_SIZE, IMG_SIZE)
        return samples

    def generate_from_z(self, z):

        self.eval()
        with torch.no_grad():
            samples = self.decode(z)  # (num_samples, CHANNELS, IMG_SIZE, IMG_SIZE)
        return samples

    def load_models(self):

        model_path = os.path.join(self.result_dir, "vae_best.pth")

        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            print("[INFO] Loaded best VAE weights.")
        else:
            print("[INFO] No VAE checkpoint found.")

    def train_vae(self, epoch, dataloader, loss_fn):
        self.train()
        train_loss = 0
        for data in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, logvar = self.forward(data)
            loss = loss_fn(recon_x=recon, x=data, mu=mu, logvar=logvar)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            #scheduler.step(loss)

        train_loss /= len(dataloader.dataset)
        print(f'Train Epoch: {epoch} | Loss: {train_loss:.4f}')
        return train_loss

    def validate(self, epoch, dataloader, loss_fn):
        self.eval()
        validate_loss = 0
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                recon, mu, logvar = self.forward(data)
                validate_loss += loss_fn(recon_x=recon, x=data, mu=mu, logvar=logvar).item()

        validate_loss /= len(dataloader.dataset)
        print(f'Validate Epoch: {epoch} | Loss: {validate_loss:.4f}')
        return validate_loss

    # Wizualizacja rekonstrukcji
    def visualize_reconstruction(self, epoch, dataloader):
        self.eval()
        with torch.no_grad():
            sample = next(iter(dataloader)).to(self.device)

            recon, mu, logvar = self.forward(sample)

            #recon = (recon + 1) / 2

            fig, axes = plt.subplots(2, 16, figsize=(32, 8))
            for i in range(16):
                axes[0, i].imshow((sample[i]).cpu().permute(1, 2, 0))
                axes[0, i].axis('off')
                axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0))
                axes[1, i].axis('off')
            plt.savefig(f'{self.result_dir}/recon_{epoch % 100}.png')
            plt.close()

    def generate_new_data(self, num_samples=50, output_dir="generated_images/CNNVAE"):
        """
        Generuje i zapisuje obrazy z losowego N(mu, exp(logvar)), gdzie mu i logvar ~ N(0,1)

        Args:
            num_samples (int): liczba obrazów
            device (str): urządzenie
        """

        z = torch.randn(num_samples, LATENT_DIM).to(self.device)

        # Wygeneruj obrazy
        images = self.generate_from_z(z)

        # Zapisz obrazy
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(images):
            vutils.save_image(img, os.path.join(output_dir, f"generated_{i}.png"))

    def generate_similar_data(self, data, num_samples=50, output_dir="generated_images/CNNVAE"):
        """
        Generuje i zapisuje obrazy z losowego N(mu, exp(logvar)), gdzie mu i logvar ~ N(0,1)

        Args:
            num_samples (int): liczba obrazów
            device (str): urządzenie
            :param data: dane wejściowe
        """

        # Wybieramy próbki z danych
        small_data = Subset(data, range(num_samples))

        small_images = []
        for idx in range(len(small_data)):
            image = small_data[idx]  # Rozpakuj obraz i etykietę
            small_images.append(image)

        # Łączymy obrazy w jeden tensor
        small_images_tensor = torch.stack(small_images).to(self.device)  # Przenosimy na odpowiednie urządzenie

        # Uzyskujemy mu i logvar
        mu, logvar = self.encode(small_images_tensor)

        logvar *= 2 # Parametr wariancji do wyboru

        # Reparametryzacja
        z = self.reparameterize(mu, logvar)

        # Generowanie obrazów
        images = self.generate_from_z(z)

        # Zapisujemy wygenerowane obrazy
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(images):
            vutils.save_image(img, os.path.join(output_dir, f"generated_{i}.png"))
