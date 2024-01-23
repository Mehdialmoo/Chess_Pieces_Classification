"""
DocString
"""
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


class ChessDB(Dataset, pl.LightningDataModule):
    """
    DocString
    """

    def __init__(self, directory, transform, batch_size) -> None:
        """DocString"""
        super().__init__()
        self.dir = directory
        self.transform = transform
        self.batch_size = batch_size
        self.labels = os.listdir(self.dir)
        self.dir_list = [os.path.join(self.dir, path) for path in self.labels]
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
            dataset = ImageFolder(root=self.dir, transform=self.transform)
            n_data = len(dataset)
            no_train = round(train_ratio * n_data)
            no_valid = round(valid_ratio * n_data)
            no_test = round(test_ratio * n_data)

            print(f"""Splited data :
                  tain sample number:{no_train}
                  validation sample number:{no_valid}
                  test sample number:{no_test}""")

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
            fig, ax = plt.subplots(figsize=(2, 5))

            ax.bar("DB", no_train, width=0.2,
                   label="Train", color="navy")
            ax.bar("DB", no_valid, width=0.2,
                   label="Validation", bottom=no_train, color="steelblue")
            ax.bar("DB", no_test, width=0.2,
                   label="Test", bottom=no_train+no_valid, color="royalblue")

            # Set the y-axis limit and add labels for the x and y axes
            ax.set_ylim(0, 1500)
            plt.ylabel("Image Count")
            plt.xlabel("Class")
            plt.title("Chess Piece Image Count")
            ax.legend()
        else:
            print("invalid ratio")

    def train_dataloader(self):
        return self.train_dataset

    def valid_dataloader(self):
        return self.valid_dataset

    def test_dataloader(self):
        return self.test_dataset
