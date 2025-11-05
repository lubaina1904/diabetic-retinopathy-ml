"""
MODEL MODULE - Neural Network Architecture

This module defines the deep learning model for diabetic retinopathy classification.

KEY CONCEPT: Transfer Learning
- Use pretrained model (trained on ImageNet)
- Replace final layer for our 5-class problem
- Much faster and better than training from scratch
"""

import torch
import torch.nn as nn
import timm  # PyTorch Image Models library


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy Classifier using Transfer Learning

    Architecture:
    Input (224×224×3) → EfficientNet Backbone → Features → Classifier → 5 classes
    """

    def __init__(self, model_name='efficientnet_b0', num_classes=5, pretrained=True):
        """
        Initialize the model

        Args:
            model_name: Which pretrained model to use (e.g., 'efficientnet_b0')
            num_classes: Number of output classes (5 for DR: 0-4)
            pretrained: Whether to load ImageNet weights
        """
        super(DRClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pretrained backbone
        # num_classes=0 means remove the classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove original classifier
        )

        # Get number of features from backbone
        self.num_features = self.backbone.num_features

        # Create our custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # Prevent overfitting
            nn.Linear(self.num_features, num_classes)
        )

        print("Created {} model".format(model_name))
        print("Feature dimension: {}".format(self.num_features))
        print("Output classes: {}".format(num_classes))

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            logits: Raw scores [batch_size, num_classes]
        """
        # Extract features using backbone
        features = self.backbone(x)  # [batch_size, num_features]

        # Classify
        logits = self.classifier(features)  # [batch_size, num_classes]

        return logits


def create_model(model_name='efficientnet_b0', num_classes=5, pretrained=True):
    """
    Factory function to create a model
    
    This is a convenient way to create models with default settings
    
    Args:
        model_name: Backbone model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        DRClassifier: Initialized model
    """
    model = DRClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def count_parameters(model):
    """
    Count total and trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: {:,}".format(total))
    print("Trainable parameters: {:,}".format(trainable))

    return total, trainable

