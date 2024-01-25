import torch
import torch.nn as nn
import torch.nn.functional as func
from pytorch_lightning import LightningModule


class ConvolutionalNetwork(LightningModule):
    """This code defines a convolutional neural network using PyTorch
    and PyTorch Lightning.
    The network consists of two convolutional layers,
    by four fully connected layers."""
    def __init__(self, labels):
        super(ConvolutionalNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(in_features=16 * 54 * 54, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=20)
        self.fc4 = nn.Linear(in_features=20, out_features=len(labels))

    def forward(self, tensor):
        """The forward method defines the forward pass of the network.
        It applies the convolutional layers, followed by max pooling,
        and then flattens the tensor to pass it through
        the fully connected layers. The final output is passed through
        a log softmax function."""
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
        """The configure_optimizers method returns
        an Adam optimizer with a specified learning rate."""
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def training_step(self, train_batch):
        """The training_step method performs one step of training."""
        tensor, lbl = train_batch
        val = self(tensor)
        loss = func.cross_entropy(val, lbl)
        pred = val.argmax(dim=1, keepdim=True)
        acc = pred.eq(lbl.view_as(pred)).sum().item() / lbl.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch):
        """The validation_step method performs one step of validation."""
        tensor, lbl = val_batch
        val = self(tensor)
        loss = func.cross_entropy(val, lbl)
        pred = val.argmax(dim=1, keepdim=True)
        acc = pred.eq(lbl.view_as(pred)).sum().item() / lbl.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch):
        """The test_step method performs one step of testing."""
        tensor, lbl = test_batch
        val = self(tensor)
        loss = func.cross_entropy(val, lbl)
        pred = val.argmax(dim=1, keepdim=True)
        acc = pred.eq(lbl.view_as(pred)).sum().item() / lbl.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)


"""for every epoch these function will be repeated and
because of that it inherites it self"""
