import sys
import unittest
from torchvision import transforms

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/../SE/vars/')

from vars.loading_data import ChessDB as DB


PATH = "./Data/Chess/"

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
    """"DocString"""
    def setUp(self) -> None:
        """"DocString"""
        self.data = DB("Data/Chess", transform=TRANS, batch_size=BATCH)

    def test_dataloading_attributes(self):
        """"DocString"""
        print(self.data.labels)
        print(len(self.data))
        print(self.data.dir_list)
        self.assertEqual(self.data.labels, 6)

    def test_dataloading(self):
        """"DocString"""
        print(self.data.dataset, self.data.dataset_label)
        self.assertEqual(len(self.data.dataset), len(self.data.dataset_label))

    def test_slpit(self):
        """Split the dataset into training and testing sets."""
        self.data.db_split(train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2)
        trainDB = self.data.train_dataloader()
        validDB = self.data.valid_dataloader()
        testDB = self.data.test_dataloader()
        s = len(trainDB)+len(validDB)+len(testDB)
        self.assertEqual(s, len(self.data))


if __name__ == "__main__":
    unittest.main()
