"""
DocString
"""
import os
import torch
import numpy as np
import splitfolders
import matplotlib.pyplot as plt


from PIL import Image
from torch.utils.data import Dataset
from  torchvision import transforms as transforms
from torch.utils.data import DataLoader


class ChessDB(Dataset):
    """
    DocString
    """

    def __init__(self, directory) -> None:
        """DocString"""
        super().__init__()
        self.dir = directory
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
                img = plt.imread(os.path.join(self.dir, subfolder, filename), 0)
                imgg.append(img)
                labels.append(subfolder)
        return imgg, labels
    
    
    def db_split(self, out_dir, train_ratio, valid_ratio, test_ratio):
            if train_ratio+valid_ratio+test_ratio == 1:
            dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)
            n_data = len(dataset)
            no_train = int(train_ratio * n_data)
            no_valid = int(valid_ratio * n_data)
            no_test = int(test_ratio * n_data)

            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [no_train, no_valid,no_test])

            self.train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.valid_dataset = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
            self.test_dataset = DataLoader(test_dataset, batch_size=self.batch_size)

            # Create a bar plot of image counts for each class in each dataset split
            fig, ax = plt.subplots(figsize = (8.5, 7))

            ax.bar(self.labels, train_image_counts, width = 0.4, label = "Train", color = "#0b090a")
            ax.bar(self.labels, val_image_counts, width = 0.4, label = "Validation", bottom = train_image_counts, color = "#ba181b")
            ax.bar(self.labels, test_image_counts, width = 0.4, label = "Test", bottom = train_image_counts + val_image_counts, color="#b0c4de")

            # Set the y-axis limit and add labels for the x and y axes
            ax.set_ylim(0, 200)
            plt.ylabel("Image Count")
            plt.xlabel("Class")
            plt.title("Chess Piece Image Count by Class")
            ax.legend()

            tarin_db = os.listdir(os.path.join(out_dir, "train"))
            val_db = os.listdir(os.path.join(out_dir, "train"))
            test_db = os.listdir(os.path.join(out_dir, "train"))



    # ------------------------------------------------

    class DataModule(pl.LightningDataModule):
    
    def __init__(self, transform=transform, batch_size=32):
        super().__init__()
        self.root_dir = "/kaggle/input/chessman-image-dataset/Chessman-image-dataset/Chess/"
        self.transform = transform
        self.batch_size = batch_size

    def split(self, train_ratio, valid_ratio, test_ratio):


    def train_dataloader(self):
        return self.train_dataset

    def valid_dataloader(self):
        return self.valid_dataset

    def test_dataloader(self):
        return self.test_dataset