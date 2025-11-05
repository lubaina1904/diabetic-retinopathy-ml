"""
TRAINING MODULE - Handles centralized model training and evaluation

This module provides the Trainer class for centralized (baseline) training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import os


class Trainer:
    """
    Handles training and evaluation of models for centralized learning
    """

    def __init__(self, model, train_loader, val_loader, device='cuda', learning_rate=0.001):
        """
        Initialize trainer

        Args:
            model: The neural network
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_kappa': []
        }

        self.best_val_kappa = 0.0

        print("Trainer initialized")
        print("Device: {}".format(device))
        print("Learning rate: {}".format(learning_rate))

    def train_epoch(self):
        """
        Train for one epoch

        Returns:
            tuple: (Average loss, Average accuracy) for this epoch
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': '{:.3f}'.format(running_loss / (pbar.n + 1)),
                'acc': '{:.2f}%'.format(100. * correct / total)
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """
        Validate on validation set

        Returns:
            tuple: (Loss, accuracy, kappa score)
        """
        self.model.eval()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

        kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')

        return val_loss, val_acc, kappa

    def train(self, num_epochs, save_dir='results'):
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints

        Returns:
            dict: Training history
        """
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*60)
        print("Starting training for {} epochs".format(num_epochs))
        print("="*60 + "\n")

        for epoch in range(1, num_epochs + 1):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print("-" * 40)

            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc, val_kappa = self.validate()

            # Update learning rate based on validation loss
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # Print if LR changed
            if new_lr != old_lr:
                print("\nLearning rate reduced: {:.6f} -> {:.6f}".format(old_lr, new_lr))

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_kappa'].append(val_kappa)

            print("\nEpoch {} Summary:".format(epoch))
            print("  Train Loss: {:.4f} | Train Acc: {:.2f}%".format(train_loss, train_acc))
            print("  Val Loss: {:.4f} | Val Acc: {:.2f}% | Kappa: {:.4f}".format(
                val_loss, val_acc, val_kappa))

            if val_kappa > self.best_val_kappa:
                self.best_val_kappa = val_kappa
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_kappa': val_kappa,
                    'val_acc': val_acc
                }, save_path)
                print("  Saved best model (Kappa: {:.4f})".format(val_kappa))

            print("="*60 + "\n")

        print("Training complete!")
        print("Best validation kappa: {:.4f}".format(self.best_val_kappa))

        return self.history


def evaluate_model(model, dataloader, device):
    """
    Comprehensive evaluation with confusion matrix

    Args:
        model: Trained model
        dataloader: DataLoader for test data
        device: 'cuda' or 'cpu'

    Returns:
        dict: Dictionary with metrics (accuracy, kappa, confusion_matrix, etc.)
    """
    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating model...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print("Accuracy: {:.2f}%".format(accuracy))
    print("Kappa Score: {:.4f}".format(kappa))
    print("\nConfusion Matrix:")
    print(cm)

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names))

    results = {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }

    return results

