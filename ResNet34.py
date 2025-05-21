import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

class ResNetTrainer:
    def __init__(self,
                 model_name: str = 'resnet34',
                 num_classes: int = 2,
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 step_size: int = 7,
                 gamma: float = 0.1,
                 device: torch.device = None):
        """
        Initializes the ResNetTrainer.

        Args:
            model_name (str): Name of torchvision ResNet model to use (e.g., 'resnet18', 'resnet50').
            num_classes (int): Number of output classes.
            lr (float): Learning rate for optimizer.
            momentum (float): Momentum for SGD.
            step_size (int): Epoch step size for LR scheduler.
            gamma (float): LR decay factor for scheduler.
            device (torch.device): Device to run on; defaults to CUDA if available.
        """
        # Device setup
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # Load pretrained ResNet
        self.model = getattr(models, model_name)(weights='IMAGENET1K_V1')
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        self.model = self.model.to(self.device)

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # Tracking
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train(self,
              train_loader,
              val_loader,
              num_epochs: int = 25,
              save_best_to: str = 'best_model.pth'):
        """
        Trains the model and validates at each epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            num_epochs (int): Number of training epochs.
            save_best_to (str): Path to save the best model by validation accuracy.
        """
        best_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)

            # Validation phase
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Step scheduler
            self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(save_best_to)

    def validate(self, loader):
        """
        Validates the model on the given DataLoader.

        Args:
            loader (DataLoader): DataLoader for validation or test data.

        Returns:
            loss (float): Average loss over the dataset.
            acc (float): Accuracy over the dataset.
        """
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

        return running_loss / total, running_corrects / total

    def visualize(self):
        """
        Plots training and validation loss and accuracy over epochs.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.figure()
        plt.plot(epochs, self.history['train_loss'], label='Train Loss')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over epochs')
        plt.show()

        plt.figure()
        plt.plot(epochs, self.history['train_acc'], label='Train Acc')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over epochs')
        plt.show()

    def save_model(self, path: str):
        """
        Saves the current state_dict of the model.

        Args:
            path (str): File path to save the model weights.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """
        Loads model weights from a given file into the model architecture.

        Args:
            path (str): File path from which to load the weights.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
