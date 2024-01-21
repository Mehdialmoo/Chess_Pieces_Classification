import sys
import unittest
from torchvision import transforms




# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/../SE/vars/')

from vars.loading_data import ChessDB as DB
from vars.utilities import *


PATH = "./Data/Chess/"
CSV_PATH = "./Data/"
BATCH = 32

transform=transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])


class Testloading_data(unittest.TestCase):
    """"DocString"""
    def setUp(self) -> None:
        self.data = DB("Data/Chess")

    def test_functions(self):
        pass

if __name__ == "__main__":
    unittest.main()
