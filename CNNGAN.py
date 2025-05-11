import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, IMG_SIZE=128,
                 CHANNELS=3,
                 LATENT_DIM=256):
        super(Generator, self).__init__()
        self.img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)

        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, IMG_SIZE=128,
                 CHANNELS=3):
        super(Discriminator, self).__init__()
        self.img_shape = (CHANNELS, IMG_SIZE, IMG_SIZE)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(CHANNELS, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )


        # The height and width of downsampled image
        ds_size = IMG_SIZE // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity



class GANLoss(nn.Module):
    def __init__(self, loss_type="bce", device="cpu"):
        super(GANLoss, self).__init__()
        self.device = device

        if loss_type == "bce":
            self.loss = nn.BCELoss(reduction="sum")
        elif loss_type == "mse":
            self.loss = nn.MSELoss(reduction="sum")
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def get_target_tensor(self, prediction, target_is_real):
        """
        Creates target label tensor with the same size as the input prediction.
        """
        if target_is_real:
            return torch.ones_like(prediction, device=self.device)
        else:
            return torch.zeros_like(prediction, device=self.device)

    def forward(self, prediction, target):
        return self.loss(prediction, target)


class CNNGAN(nn.Module):

    def __init__(self, IMG_SIZE=128, CHANNELS=3, LATENT_DIM=256, device="cpu", result_dir="results", load_pretrained=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = Generator(IMG_SIZE, CHANNELS, LATENT_DIM).to(device)
        self.discriminator = Discriminator(IMG_SIZE, CHANNELS).to(device)
        self.gan_loss = GANLoss(loss_type="bce", device=device)
        self.LATENT_DIM = LATENT_DIM
        self.CHANNELS = CHANNELS
        self.IMG_SIZE = IMG_SIZE
        self.device = device

        self.result_dir = result_dir

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

        if load_pretrained:
            self.load_models()

    def train_gan(self, epoch, dataloader):
        # Ustawienie modeli w tryb treningowy
        self.generator.train()
        self.discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for data in dataloader:
            # Przenosimy dane na odpowiednie urządzenie
            real_imgs = data.to(self.device)

            # Adwersarze - Tworzymy etykiety dla prawdziwych i fałszywych obrazów
            valid = torch.ones(real_imgs.size(0), 1).to(self.device)  # Prawdziwe obrazy
            fake = torch.zeros(real_imgs.size(0), 1).to(self.device)  # Fałszywe obrazy

            # -----------------------------------
            # Trenowanie dyskryminatora
            # -----------------------------------
            self.d_optimizer.zero_grad()

            # Trening dyskryminatora na prawdziwych obrazach
            real_loss = self.gan_loss(self.discriminator(real_imgs), valid)

            # Generowanie fałszywych obrazów z generatora
            z = torch.randn(real_imgs.size(0), self.LATENT_DIM).to(self.device)  # Losowy wektor z przestrzeni latentnej
            fake_imgs = self.generator(z)

            # Trening dyskryminatora na fałszywych obrazach
            fake_loss = self.gan_loss(self.discriminator(fake_imgs.detach()), fake)

            # Łączna strata dyskryminatora
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.d_optimizer.step()

            running_d_loss += d_loss.item()

            # -----------------------------------
            # Trenowanie generatora
            # -----------------------------------
            self.g_optimizer.zero_grad()

            # Generator stara się oszukać dyskryminator
            g_loss = self.gan_loss(self.discriminator(fake_imgs), valid)
            g_loss.backward()
            self.g_optimizer.step()

            running_g_loss += g_loss.item()

        # Średnia strata dla generatora i dyskryminatora
        avg_g_loss = running_g_loss / len(dataloader)
        avg_d_loss = running_d_loss / len(dataloader)

        print(f"Epoch {epoch}: Generator Loss = {avg_g_loss:.4f}, Discriminator Loss = {avg_d_loss:.4f}")

        return avg_g_loss, avg_d_loss

    def validate(self, epoch, dataloader):
        # Ustawienie modeli w tryb ewaluacji (bez gradientów)
        self.generator.eval()
        self.discriminator.eval()

        running_loss = 0.0

        with torch.no_grad():
            for data in dataloader:
                real_imgs = data.to(self.device)

                valid = torch.ones(real_imgs.size(0), 1).to(self.device)  # Prawdziwe obrazy
                fake = torch.zeros(real_imgs.size(0), 1).to(self.device)  # Fałszywe obrazy

                # Przewidywanie na prawdziwych obrazach
                real_loss = self.gan_loss(self.discriminator(real_imgs), valid)

                # Generowanie fałszywych obrazów
                z = torch.randn(real_imgs.size(0), self.LATENT_DIM).to(self.device)
                fake_imgs = self.generator(z)

                # Przewidywanie na fałszywych obrazach
                fake_loss = self.gan_loss(self.discriminator(fake_imgs), fake)

                # Łączna strata
                total_loss = (real_loss + fake_loss) / 2
                running_loss += total_loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}: Validation Loss = {avg_loss:.4f}")

        return avg_loss

    def visualize_reconstruction(self, epoch):

        num_samples = 16

        # Generowanie próbek
        z = torch.randn(num_samples, self.LATENT_DIM).to(self.device)  # Generowanie z losowego wektora latentnego
        fake_imgs = self.generator(z)

        fake_imgs = (fake_imgs + 1) / 2

        rows = 4  # Zaokrąglamy liczbę wierszy
        cols = 4  # Liczba kolumn

        # Tworzymy siatkę wykresów
        fig, axes = plt.subplots(rows, cols, figsize=(8, 8))

        # Pętla przez obrazy
        for i in range(rows):
            for j in range(cols):
                index = i + j  # Indeks obrazu
                if index < num_samples:  # Sprawdzamy, czy mamy obraz do wyświetlenia
                    axes[i, j].imshow(fake_imgs[index].detach().cpu().numpy().transpose(1, 2,
                                                                                        0))  # Przekształcamy (C, H, W) na (H, W, C)
                axes[i, j].axis('off')  # Ukrywamy osie

        plt.savefig(f'{self.result_dir}/recon_{epoch%100}.png')
        plt.close(fig)  # Zamykamy wykres, aby zwolnić pamięć


    def load_models(self):
        generator_path = os.path.join(self.result_dir, "best_generator.pth")
        discriminator_path = os.path.join(self.result_dir, "best_discriminator.pth")

        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print("[INFO] Loaded best generator weights.")
        else:
            print("[INFO] No generator checkpoint found.")

        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            print("[INFO] Loaded best discriminator weights.")
        else:
            print("[INFO] No discriminator checkpoint found.")

    def save_generator(self):
        torch.save(self.generator.state_dict(), f'{self.result_dir}/best_generator.pth')

    def save_discriminator(self):
        torch.save(self.discriminator.state_dict(), f'{self.result_dir}/best_discriminator.pth')
