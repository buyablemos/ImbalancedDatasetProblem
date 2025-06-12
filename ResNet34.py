import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, balanced_accuracy_score, recall_score, confusion_matrix



class ResNetTrainer:
    def __init__(self,
                 model_name: str = 'resnet34',
                 num_classes: int = 2,
                 lr: float = 1e-4,
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

        # Unfreeze last 2 layer
        for name, param in self.model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Loss, optimizer, scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        # Tracking
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}



    def train(self,
              train_loader,
              num_epochs: int = 25):
        """
        Trains the model and validates at each epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of training epochs.
        """
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        for epoch in range(1, num_epochs + 1):
            # Training phase
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                inputs_resnet = (inputs - imagenet_mean) / imagenet_std

                self.optimizer.zero_grad()
                outputs = self.model(inputs_resnet)
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

            # Step scheduler
            self.scheduler.step()

            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 📊 Zapisz wykres strat i dokładności po zakończeniu treningu
        epochs_range = range(1, num_epochs + 1)

        plt.figure(figsize=(8, 5))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history['train_loss'], label='Train Loss', marker='o')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("charts/resnet_training_curves.png")
        plt.close()




    def validate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                inputs_resnet = (inputs - imagenet_mean) / imagenet_std

                outputs = self.model(inputs_resnet)

                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # F2 score (beta=2)
        f2 = fbeta_score(all_labels, all_preds, beta=2, average='binary')

        # Balanced accuracy
        bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Recall (czułość)
        recall = recall_score(all_labels, all_preds, average='binary')

        # Specificity (swoistość) = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp)

        return f2, bal_acc, recall, specificity

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