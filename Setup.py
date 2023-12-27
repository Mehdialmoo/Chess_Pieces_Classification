""" Because of the loading problems decided to copy the exact values from (https://www.kaggle.com/code/minniekabra/pascal-voc-loaded-images/notebook) cause i had problems with loading the data"""

import cv2
import torch
import albumentations as alb


from albumentations.pytorch import ToTensorV2

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


scale = 1.1
train_transforms = alb.Compose(
    [
        alb.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        alb.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        alb.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        alb.ColorJitter(brightness=0.6, contrast=0.6,
                        saturation=0.6, hue=0.6, p=0.4),
        alb.OneOf(
            [
                alb.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                alb.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        alb.HorizontalFlip(p=0.5),
        alb.Blur(p=0.1),
        alb.CLAHE(p=0.1),
        alb.Posterize(p=0.1),
        alb.ToGray(p=0.1),
        alb.ChannelShuffle(p=0.05),
        alb.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=alb.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = alb.Compose(
    [
        alb.LongestMaxSize(max_size=IMAGE_SIZE),
        alb.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        alb.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=alb.BboxParams(
        format="yolo", min_visibility=0.4, label_fields=[]),
)

PASCAL_CLASSES = []

COCO_LABELS = []