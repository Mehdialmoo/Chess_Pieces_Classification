import torch
import vars.loading_data as loading_data
import pytorch_lightning as pl

from vars.model import ConvolutionalNetwork
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from vars.wandb_logger import logger

# ========================================
# from vars.my_wandb_logger import logger
# ========================================


class model_run():
    """this class makes it easier for non developers to setup
    the data and model,train and test a convolutional neural network on
    dataset and to be able to retrive pretrained models as checkpoints"""
    def __init__(self, data_directory, transformer,
                 batch_size, ratio, epoch, learning_rate) -> None:
        """ creates an object to save all the attribuites we need
        to train the model"""
        self._dir = data_directory
        self._epoch = epoch
        self._trans = transformer
        self._batch_sz = batch_size
        self._ratio = ratio
        self._lr = learning_rate
        self.ES = EarlyStopping(monitor="train_loss", mode="min", patience=3)

    def setup_data(self):
        # this function uses loading_data file  to load and splite the data
        self.data = loading_data.ChessDB(directory=self._dir,
                                         transform=self._trans,
                                         batch_size=self._batch_sz)
        self.data.db_split(train_ratio=self._ratio[0],
                           valid_ratio=self._ratio[1],
                           test_ratio=self._ratio[2])
        self.train_DB = self.data.train_dataloader()
        self.validation_DB = self.data.valid_dataloader()
        self.test_DB = self.data.test_dataloader()

    def setup_model(self):
        """create a new instance of our convolutional network and
        checks if its avalible to be run over gpu"""

        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvolutionalNetwork(labels=self.data.labels)
        self.model.configure_optimizers(learning_rate=self._lr)

    def load_model(self, checkpoint_dir):
        # loads in weights from a previous training session
        self.model = ConvolutionalNetwork.load_from_checkpoint(
            checkpoint_path=checkpoint_dir, labels=self.data.labels)

    def load_full(self, checkpoint_dir):
        # loads in everything including optimizer and scheduler
        self.model = ConvolutionalNetwork(labels=self.data.labels)
        trainer = pl.Trainer(max_epochs=self._epoch,
                             accelerator='gpu',
                             callbacks=self.ES,
                             default_root_dir="./")
        # automatically restores model, epoch, step, LR schedulers, etc...
        trainer.fit(self.model, self.train_DB, ckpt_path=checkpoint_dir)

    def train_run(self):
        """trains the model on traning set of data"""
        """
        self.trainer = pl.Trainer(max_epochs=self._epoch,
                                  accelerator="cpu",
                                  default_root_dir="./checkpoints/")

        # for training on cpu
        """

        """
        self.trainer = pl.Trainer(max_epochs=self._epoch,
                                  logger=logger(),
                                  accelerator="gpupu",
                                  default_root_dir="./checkpoints/")

        # for training on gpu and use of  wandb logger
        """

        self.trainer = pl.Trainer(max_epochs=self._epoch,
                                  accelerator='gpu',
                                  callbacks=self.ES,
                                  default_root_dir="./")  # for training on gpu
        # if there is a need to see the logs on wand be add logger=logger()
        self.model.setup(stage='fit')
        self.trainer.fit(self.model, train_dataloaders=self.train_DB)

    def validation_run(self):
        """validates the current state of the model
        against the validation set"""
        self.data.setup(stage='validate')
        self.trainer.validate(dataloaders=self.validation_DB, ckpt_path='best')

    def test_run(self):
        """tests the trained model against the testing set"""
        self.data.setup(stage='test')
        self.trainer.test(dataloaders=self.test_DB, ckpt_path='best')

    def evaluation(self):
        """evaluate the performance of the model
        using metrics defined during initialization"""
        self.model.eval()
        ground_truth = []
        model_outputs = []
        with torch.no_grad():
            for item in self.test_DB:
                item_img, item_lbl = item[0], item[1]
                output = self.model(item_img).argmax(dim=1)
                for i in range(len(output)):
                    ground_truth.append(item_lbl[i].item())
                    model_outputs.append(output[i].item())
        print(classification_report(ground_truth,
                                    model_outputs,
                                    target_names=self.data.labels, digits=4))
