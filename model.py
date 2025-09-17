# backend/model.py

import torch
import torch.nn as nn
import torchvision.models as models

# Base class for image classification models
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = nn.functional.cross_entropy(out, labels)
        acc = (torch.argmax(out, dim=1) == labels).float().mean()
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_loss = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

# ResNet34-based model for Plant Disease Detection
class Plant_Disease_Model2(ImageClassificationBase):
    def __init__(self):
        super(Plant_Disease_Model2, self).__init__()
        # Note: pretrained is set to False here because we assume you trained and saved the model state.
        self.network = models.resnet34(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        return self.network(xb)
