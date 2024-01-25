import sys
import unittest
from torchvision import transforms

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/../SE/vars/')

from vars.loading_data import ChessDB as DB
from vars.utilities import create_CSV, plot_bar, plot_img, pre_process


PATH = "./Data/Chess/"
CSV_PATH = "./Data/"
BATCH = 32

TRANS = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize to 224*224 pixels
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])


class Testloading_data(unittest.TestCase):
    """DocString"""
    def setUp(self) -> None:
        """DocString"""
        self.data = DB("Data/Chess", transform=TRANS, batch_size=BATCH)

    def test_create_CSV(self):
        """DocString"""
        create_CSV(dir=self.data.dir, out_dir=CSV_PATH)

    def test_plot_bar(self):
        """DocString"""
        plot_bar(self.data.dir, self.data.labels)

    def test_plot_img(self):
        """DocString"""
        plot_img(dir_list=self.data.dir_list, labels=self.data.labels)

    def test_pre_process(self):
        """DocString"""
        pre_process(dir_list=self.data.dir_list)


if __name__ == '__main__':
    unittest.main()
