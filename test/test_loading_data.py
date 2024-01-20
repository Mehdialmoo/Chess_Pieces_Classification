import sys
import unittest


# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/../SE/vars/')
import loading_data


class Testloading_data(unittest.TestCase):
    """"DocString"""
    def setUp(self) -> None:
        self.data = ChessDB("Data/Chess")

    def test_functions(self):
        """"DocString"""
        print(self.data.dir)
        print(self.data.dir_list)
        print(self.data.labels)
        a = self.data.filecounter()
        print (a)
        print(a .keys())
        print(a.values())
        self.data.plot_bar()
        img, label = self.data.data_loader()
        print(img[0][12].shape)
        print(label[100])
        print(len(img))
        self.data.plot_img()
        self.data.pre_process()
        self.data.db_split("Data/ready_data")


if __name__ == "__main__":
    unittest.main()
