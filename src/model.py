
import torch
import torch.nn as nn
import timm  


class DRClassifier(nn.Module):
    

    def __init__(self, model_name='efficientnet_b0', num_classes=5, pretrained=True):
        
        super(DRClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  
        )

        self.num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  
            nn.Linear(self.num_features, num_classes)
        )

        print("Created {} model".format(model_name))
        print("Feature dimension: {}".format(self.num_features))
        print("Output classes: {}".format(num_classes))

    def forward(self, x):
        features = self.backbone(x)  
        logits = self.classifier(features) 
        return logits


def create_model(model_name='efficientnet_b0', num_classes=5, pretrained=True):
    model = DRClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters: {:,}".format(total))
    print("Trainable parameters: {:,}".format(trainable))

    return total, trainable

