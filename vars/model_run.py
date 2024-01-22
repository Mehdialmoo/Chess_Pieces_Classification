import torch
from torchvision import transforms
import vars.loading_data as loading_data
from vars.model import ConvolutionalNetwork
import pytorch_lightning as pl
from sklearn.metrics import classification_report

from pytorch_lightning.loggers import WandbLogger
import wandb
wandb.login(key="f2dff4d9b4135a7933dfe4796a30db28811a5c5e")


class  model_run():
    """DocString"""
    def __init__(self, data_directory, transformer, batch_size, ratio, epoch) -> None:
        self.data = loading_data.ChessDB(directory=data_directory,
                                         transform=transformer,
                                         batch_size=batch_size)
        self.data.db_split(train_ratio=ratio[0],
                           valid_ratio=ratio[1], test_ratio=ratio[2])
        self.train_DB = self.data.train_dataloader()
        self.validation_DB = self.data.valid_dataloader()
        self.test_DB = self.data.test_dataloader()
        self.epoch = epoch

    def setup(self):
        self.logger = WandbLogger(log_model="all")
        self.model = ConvolutionalNetwork(labels=self.data.labels)
        self.model.configure_optimizers(learning_rate=0.1)

    def train_run(self):
        self.trainer = pl.Trainer(max_epochs=self.epoch,
                                  logger=self.logger, accelerator="gpu")
        self.model.setup(stage='fit')
        self.trainer.fit(self.model, self.train_DB)

    def validation_run(self):
        self.data.setup(stage='validate')
        self.trainer.validate(dataloaders=self.validation_DB)

    def test_run(self):
        self.data.setup(stage='test')
        self.trainer.test(dataloaders=self.test_DB)

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
        print(classification_report(ground_truth, model_outputs, target_names=self.data.labels, digits=4))
