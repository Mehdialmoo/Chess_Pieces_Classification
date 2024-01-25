import unittest
import torch

from torchvision import transforms
from vars.model_run import model_run

PATH = "./Data/Chess/"
CSV_PATH = "./Data/"
CHECK_PATH = "./lightning_logs/version_1/checkpoints/epoch=5-step=204.ckpt"
BATCH = 20
RATIO = [0.6, 0.2, 0.2]
EPOCH = 5
LR = 0.001

TRANS = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])


class Testmodelrun(unittest.TestCase):
    """DocString"""
    def setUp(self) -> None:
        self.mode = model_run(data_directory=PATH,
                               transformer=TRANS,
                               batch_size=BATCH, ratio=RATIO,
                               epoch=EPOCH,
                               learning_rate=LR)
        self.mode.setup_data()
        self.mode.setup_model()

    def test_train_run(self):
        """DocString"""
        self.mode.train_run()
        self.assert_("Runs Successfully")

    def test_validation_run(self):
        """DocString"""
        self.mode.validation_run()
        self.assert_("Runs Successfully")

    def test_test_run(self):
        """DocString"""
        self.mode.test_run()
        self.assert_("Runs Successfully")

    def test_evaluation(self):
        """DocString"""
        self.mode.evaluation()
        self.assert_("Runs Successfully")

    def test_load_full(self):
        """DocString"""
        self.mode = self.mode.load_full(checkpoint_dir=CHECK_PATH)
        self.assert_("Runs Successfully")

    def test_load_model(self):
        """DocString"""
        self.model = self.mode.load_model(checkpoint_dir=CHECK_PATH)
        self.assert_("Runs Successfully")


if __name__ == '__main__':
    unittest.main()
