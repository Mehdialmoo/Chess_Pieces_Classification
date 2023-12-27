import os
import numpy as np
import pandas as pd

import torch

from PIL import Image
from PIL import ImageFile
from torch.utils.data import dataset
from torch.utils.data import datasetLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataSet():
    """ """

    def __init__(self,
                 csv_file,
                 image_dir,
                 label_dir,
                 anchors,
                 image_size,
                 transform=None) -> None:
        """ """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = image_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.anchors = torch.tensor(
            anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

    def __len__(self) -> len:  # editing dunder methods
        """ """
        return len(self.annotations)

    def __getitem__(self, idx) -> None:
        """ """
        label_path = os.path.join(
            self.label_dir, self.annotations.iloc[idx, 1])
        bounding_boxes = np.roll(
            np.loadtxt(fname=label_path, delimiter=" ", ndmin=2),
            shift=4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # condition checker

        if self.transform is True:
            augmentations = self.transform(image=image, bbox=bounding_boxes)
            image = augmentations["image"]
            bounding_boxes = augmentations["bounding_boxes"]

        # (as: YOLOv1.PDF) below assumes 3 scale predictions  and same number of anchors per scale
