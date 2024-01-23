import torch
import vars.loading_data as loading_data
from vars.model import ConvolutionalNetwork
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# ========================================
# from vars.my_wandb_logger import logger
# ========================================


class model_run():
    """DocString"""
    def __init__(self, data_directory, transformer,
                 batch_size, ratio, epoch, learning_rate) -> None:
        self._dir = data_directory
        self._epoch = epoch
        self._trans = transformer
        self._batch_sz = batch_size
        self._ratio = ratio
        self._lr = learning_rate

    def setup_data(self):
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
        # self.logger = logger()
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvolutionalNetwork(labels=self.data.labels)
        self.model.configure_optimizers(learning_rate=self._lr)

    def load_model(self, checkpoint_dir):
        self.model = ConvolutionalNetwork.load_from_checkpoint(
            checkpoint_path=checkpoint_dir, labels=self.data.labels)

    def train_run(self):
        """
        self.trainer = pl.Trainer(max_epochs=self._epoch,
                                  logger=logger(),
                                  accelerator=acc,
                                  default_root_dir="./checkpoints/")
        """
        EarlyStop = EarlyStopping(monitor="train_loss", mode="min", patience=3)
        self.trainer = pl.Trainer(max_epochs=self._epoch,
                                  accelerator='gpu',
                                  callbacks=EarlyStop,
                                  default_root_dir="./checkpoints/")
        self.model.setup(stage='fit')
        self.trainer.fit(self.model, self.train_DB)

    def validation_run(self):
        self.data.setup(stage='validate')
        self.trainer.validate(dataloaders=self.validation_DB, ckpt_path='best')

    def test_run(self):
        self.data.setup(stage='test')
        self.trainer.test(dataloaders=self.test_DB, ckpt_path='best')

    def evaluation(self):
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
