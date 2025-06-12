import os
import shutil

from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.transforms import transforms
from CNNVAE import CNNVAE
from DatasetClasses import NegativeOnlySubset, CapsuleDataset, PositiveOnlySubset
from ResNet34 import ResNetTrainer
from CNNGAN import CNNGAN


class CNNVAEResNetEstimator(BaseEstimator):
    def __init__(self, dataset, device='cpu',
                 mu_multiplier=1.0, logvar_multiplier=1.0,
                 multiplier_generated_samples=50,
                 batch_size=16,
                 classifier_epochs=5,
                 oversampler_epochs=10,
                 vae_model=CNNVAE(),
                 output_dir="generated_images/CNNVAE",
                 IMG_SIZE=128):
        self.trainer = None
        self.classifier_dataloader = None
        self.train_neg_only_loader = None
        self.vae_model = vae_model
        self.dataset = dataset
        self.device = device
        self.mu_multiplier = mu_multiplier
        self.logvar_multiplier = logvar_multiplier
        self.multiplier_generated_samples = multiplier_generated_samples
        self.batch_size = batch_size
        self.oversampler_epochs = oversampler_epochs
        self.output_dir = output_dir
        self.classifier_epochs = classifier_epochs
        self.img_size = IMG_SIZE

    def fit(self, X=None, y=None):

        gen_dir = os.path.join(
            self.output_dir,
            f"mu_{self.mu_multiplier}_logvar_{self.logvar_multiplier}"
        )

        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)

        os.makedirs(gen_dir, exist_ok=True)

        train_ds = Subset(self.dataset, X)

        train_neg_only_dataset = NegativeOnlySubset(self.dataset, X)

        if self.multiplier_generated_samples == 'synthetic':

            num_neg = len(train_ds) - len(train_neg_only_dataset)

        else:
            num_neg = len(train_neg_only_dataset)

        self.train_neg_only_loader = DataLoader(train_neg_only_dataset, batch_size=self.batch_size, shuffle=True)

        if self.multiplier_generated_samples == 'synthetic':
            self.vae_model.generate_similar_data(
                self.train_neg_only_loader,
                mu_multiplier=self.mu_multiplier,
                log_multiplier=self.logvar_multiplier,
                num_samples=int(num_neg),
                output_dir=gen_dir
            )
        else:
            self.vae_model.generate_similar_data(
                self.train_neg_only_loader,
                mu_multiplier=self.mu_multiplier,
                log_multiplier=self.logvar_multiplier,
                num_samples=int(num_neg * self.multiplier_generated_samples),
                output_dir=gen_dir
            )

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])


        generated_dataset = CapsuleDataset(
            pos_dir=None,
            neg_dirs=[gen_dir],
            transform=transform
        )

        if self.multiplier_generated_samples == 'synthetic':
            positive_only_dataset = PositiveOnlySubset(self.dataset, X, transform=transform)
            classifier_dataset = ConcatDataset([generated_dataset, positive_only_dataset])
            self.classifier_dataloader = DataLoader(classifier_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            classifier_dataset = ConcatDataset([generated_dataset, train_ds])
            self.classifier_dataloader = DataLoader(classifier_dataset, batch_size=self.batch_size, shuffle=True)


        self.trainer = ResNetTrainer()

        self.trainer.train(
            self.classifier_dataloader,
            num_epochs=self.classifier_epochs,
        )

        return self

    def score(self, X=None, y=None):
        # Ewaluacja
        f2, bal_acc, recall, specificity = self.trainer.validate(self.classifier_dataloader)
        return f2

    def get_params(self, deep=True):
        return {
            'dataset': self.dataset,
            'device': self.device,
            'mu_multiplier': self.mu_multiplier,
            'logvar_multiplier': self.logvar_multiplier,
            'multiplier_generated_samples': self.multiplier_generated_samples,
            'batch_size': self.batch_size,
            'classifier_epochs': self.classifier_epochs,
            'oversampler_epochs': self.oversampler_epochs,
            'vae_model': self.vae_model,
            'output_dir': self.output_dir
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CNNVAEWrapper(BaseEstimator):
    def __init__(self, dataset, device='cpu',
                 batch_size=16,
                 oversampler_epochs=10):
        self.train_neg_only_loader = None
        self.vae_model = CNNVAE(device=device, result_dir='results/CNNVAE', load_pretrained=False)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.oversampler_epochs = oversampler_epochs

    def fit(self, X=None, y=None):
        train_neg_only_dataset = NegativeOnlySubset(self.dataset, X)
        self.train_neg_only_loader = DataLoader(train_neg_only_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.oversampler_epochs + 1):
            self.vae_model.train_vae(epoch=epoch, dataloader=self.train_neg_only_loader)

        return self


class CNNGANResNetEstimator(BaseEstimator):
    def __init__(self, dataset, device='cpu',
                 multiplier_generated_samples=50,
                 batch_size=16,
                 classifier_epochs=5,
                 oversampler_epochs=10,
                 gan_model=CNNGAN(),
                 output_dir="generated_images/CNNGAN",
                 IMG_SIZE=128,
                 scale_factor=1):
        self.trainer = None
        self.classifier_dataloader = None
        self.train_neg_only_loader = None
        self.gan_model = gan_model
        self.dataset = dataset
        self.device = device
        self.multiplier_generated_samples = multiplier_generated_samples
        self.batch_size = batch_size
        self.oversampler_epochs = oversampler_epochs
        self.output_dir = output_dir
        self.classifier_epochs = classifier_epochs
        self.img_size = IMG_SIZE
        self.scale_factor = scale_factor

    def fit(self, X=None, y=None):

        gen_dir = os.path.join(
            self.output_dir
        )

        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)

        os.makedirs(gen_dir, exist_ok=True)

        train_ds = Subset(self.dataset, X)

        train_neg_only_dataset = NegativeOnlySubset(self.dataset, X)

        if self.multiplier_generated_samples == 'synthetic':

            num_neg = len(train_ds) - len(train_neg_only_dataset)

        else:
            num_neg = len(train_neg_only_dataset)

        self.train_neg_only_loader = DataLoader(train_neg_only_dataset, batch_size=self.batch_size, shuffle=True)


        if self.multiplier_generated_samples == 'synthetic':
            self.gan_model.generate_new_data(
                num_samples=int(num_neg),
                output_dir=gen_dir,
                scale_factor=self.scale_factor
            )
        else:
            self.gan_model.generate_new_data(
                num_samples=int(num_neg * self.multiplier_generated_samples),
                output_dir=gen_dir,
                scale_factor=self.scale_factor
            )

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])


        generated_dataset = CapsuleDataset(
            pos_dir=None,
            neg_dirs=[gen_dir],
            transform=transform
        )

        if self.multiplier_generated_samples == 'synthetic':
            positive_only_dataset = PositiveOnlySubset(self.dataset, X, transform=transform)
            classifier_dataset = ConcatDataset([generated_dataset, positive_only_dataset])
            self.classifier_dataloader = DataLoader(classifier_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            classifier_dataset = ConcatDataset([generated_dataset, train_ds])
            self.classifier_dataloader = DataLoader(classifier_dataset, batch_size=self.batch_size, shuffle=True)


        self.trainer = ResNetTrainer()

        self.trainer.train(
            self.classifier_dataloader,
            num_epochs=self.classifier_epochs,
        )

        return self

    def score(self, X=None, y=None):
        # Ewaluacja
        f2, bal_acc, recall, specificity = self.trainer.validate(self.classifier_dataloader)
        return f2

    def get_params(self, deep=True):
        return {
            'dataset': self.dataset,
            'device': self.device,
            'multiplier_generated_samples': self.multiplier_generated_samples,
            'batch_size': self.batch_size,
            'classifier_epochs': self.classifier_epochs,
            'oversampler_epochs': self.oversampler_epochs,
            'gan_model': self.gan_model,
            'output_dir': self.output_dir
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class CNNGANWrapper(BaseEstimator):
    def __init__(self, dataset, device='cpu',
                 batch_size=16,
                 oversampler_epochs=10):
        self.train_neg_only_loader = None
        self.gan_model = CNNGAN(device=device, result_dir='results/CNNGAN', load_pretrained=False)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.oversampler_epochs = oversampler_epochs

    def fit(self, X=None, y=None):
        train_neg_only_dataset = NegativeOnlySubset(self.dataset, X)
        self.train_neg_only_loader = DataLoader(train_neg_only_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.oversampler_epochs + 1):
            self.gan_model.train_gan(epoch=epoch, dataloader=self.train_neg_only_loader)

        return self