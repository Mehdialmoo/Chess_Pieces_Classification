import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_lightning import LightningModule


class ConvolutionalNetwork(LightningModule):
    """
    DocString
    """
    def __init__(self, labels):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 54 * 54, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, len(labels))

    def forward(self, tensor):
        tensor = func.relu(self.conv1(tensor))
        tensor = func.max_pool2d(tensor, 2, 2)
        tensor = func.relu(self.conv2(tensor))
        tensor = func.max_pool2d(tensor, 2, 2)
        tensor = tensor.view(-1, 16 * 54 * 54)
        tensor = func.relu(self.fc1(tensor))
        tensor = func.relu(self.fc2(tensor))
        tensor = func.relu(self.fc3(tensor))
        tensor = self.fc4(tensor)
        return func.log_softmax(tensor, dim=1)

    def configure_optimizers(self, learning_rate=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def training_step(self, train_batch):
        X, y = train_batch
        y_hat = self(X)
        loss = func.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch):
        X, y = val_batch
        y_hat = self(X)
        loss = func.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch):
        X, y = test_batch
        y_hat = self(X)
        loss = func.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)
