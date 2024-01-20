"""
DocString
"""
import os
import torch
import matplotlib.pyplot as plt


from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder


class ChessDB(Dataset,pl.LightningDataModule):
    """
    DocString
    """

    def __init__(self, directory, transform, batch_size) -> None:
        """DocString"""
        super().__init__()
        self.dir = directory
        self.transform = transform
        self.batch_size = batch_size
        folders = os.listdir(self.dir)
        dir_list = [os.path.join(self.dir, path) for path in folders]
        self.labels = folders
        self.dir_list = dir_list
        self.data_dict = self.filecounter()
        self.dataset, self.dataset_label = self.data_loader()

    # implement __len__
    def __len__(self):
        return len(self.dataset)    

    # implement __getitem__
    def __getitem__(self, index):
        return self.dataset[index]

    def data_loader(self) -> tuple:
        """DocString"""
        imgg = []
        labels = []
        for subfolder in (os.listdir(self.dir)):
            for filename in (os.listdir(os.path.join(self.dir, subfolder))):
                img = plt.imread(
                    os.path.join(self.dir, subfolder, filename), 0)
                imgg.append(img)
                labels.append(subfolder)
        return imgg, labels

    def db_split(self, train_ratio, valid_ratio, test_ratio):
        """DocString"""
        if train_ratio+valid_ratio+test_ratio == 1:
            dataset = ImageFolder(root=self.root_dir, transform=self.transform)
            n_data = len(dataset)
            no_train = int(train_ratio * n_data)
            no_valid = int(valid_ratio * n_data)
            no_test = int(test_ratio * n_data)

            train_db, valid_db, test_db = random_split(
                dataset, [no_train, no_valid, no_test])

            self.train_dataset = DataLoader(train_db,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            self.valid_dataset = DataLoader(valid_db,
                                            batch_size=self.batch_size,
                                            shuffle=True)
            self.test_dataset = DataLoader(test_db, batch_size=self.batch_size)

            # Create a bar plot of image counts for each class
            fig, ax = plt.subplots(figsize=(8.5, 7))

            ax.bar(self.labels, no_train, width=0.4,
                   label="Train", color="red")
            ax.bar(self.labels, no_valid, width=0.4,
                   label="Validation", bottom=no_train, color="blue")
            ax.bar(self.labels, no_test, width=0.4,
                   label="Test", bottom=no_train+no_valid, color="green")

            # Set the y-axis limit and add labels for the x and y axes
            ax.set_ylim(0, 200)
            plt.ylabel("Image Count")
            plt.xlabel("Class")
            plt.title("Chess Piece Image Count by Class")
            ax.legend()
        else:
            print("invalid ratio")

    def label_loader(self):
        return self.dataset_label

    def train_dataloader(self):
        return self.train_dataset

    def valid_dataloader(self):
        return self.valid_dataset

    def test_dataloader(self):
        return self.test_dataset
