from email.utils import parsedate
import torch
import torch.nn as nn
import torch.nn.functional as F
from ShConv import ShConv
from utils import LayersHyperParameters
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import PreprocessedImageInpaintingDataset
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from torchvision import utils
from torchvision import transforms as T
from configs import training_configs, experiments_config, data_set_config, seed
import argparse

# fix the seed
seed = seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class ShepardNet(pl.LightningModule):
    def __init__(self, layers, LR):
        super(ShepardNet, self).__init__()
        self.LR = LR
        self.layers = layers
        the_input_layer = LayersHyperParameters(self.layers[0].layer_type, 3, self.layers[0].kernel_size)
        self.layers = [the_input_layer, *self.layers]

        self.modules_list = nn.ModuleList()
        for i, (input_layer, output_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            if (output_layer.layer_type == 'conv'):
                self.modules_list.append(nn.Conv2d(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
            else:
                self.modules_list.append(ShConv(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
            self.modules_list.append(nn.ReLU())
            # if (i != len(self.layers) - 1):
            #     self.modules_list.append(nn.BatchNorm2d(output_layer.kernels_num))

        # saving the hyperparameters (for wandb)
        self.save_hyperparameters()

    def forward(self, x, masks):
        for layer in self.modules_list:
            if isinstance(layer, ShConv):
                x, masks = layer(masks, x)
            else:
                x = layer(x)
        return x, masks

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        loss = F.mse_loss(original, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        test_loss = F.mse_loss(original, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        val_loss = F.mse_loss(original, x)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        return optimizer


def train(args):

    for data_path in data_set_config['paths']:

        dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])

        dataset_size = len(dataset)

        split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))

        train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))

        trainset_size = len(train_set)

        split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))

        train_set, validation_set = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))

        print(f"Training dataset contains {len(train_set)} examples\nValidation dataset conttains {len(validation_set)}\nTest dataset contains {len(test_set)}")

        batch_size = training_configs['batch_size']()
        
        train_dataloader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=6)

        validation_dataloader = DataLoader(validation_set, batch_size=batch_size,
                            shuffle=False, num_workers=3)

        test_dataloader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=2)

        # training _____________________________________________________________________________________

        for experiment in experiments_config['experiments']:

            layers = experiment['layers']

            net = ShepardNet(layers, training_configs['LR'])

            if (args.wandb_log):
                wandb_logger = WandbLogger(project="ShCNN", name=experiment['name'])
                wandb_logger.experiment.config.update({'layers': layers})
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True, logger=wandb_logger)
            else:
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True)

            trainer.fit(net, train_dataloader, validation_dataloader)
        # training _____________________________________________________________________________________


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', help="boolean value to indicate using wandb log or not.", action="store_true", default=False)
    args = parser.parse_args()
    train(args)


    # DUMMY Training
    

    # # layers = [
    # #     LayersHyperParameters('shepard', 8, 7),
    # #     LayersHyperParameters('shepard', 8, 5),
    # #     LayersHyperParameters('conv', 128, 3),
    # #     LayersHyperParameters('conv', 128, 1),
    # #     LayersHyperParameters('conv', 3, 3),
    # # ]

    # net = ShepardNet.load_from_checkpoint(r'.\ShCNN\1o1qhm8a\checkpoints\epoch=19-step=16740.ckpt')
    # net.eval()

    # dataloader_iter = iter(test_dataloader)
    # original, x, masks = next(dataloader_iter)

    # # predict with the model
    # y_hat, masks = net(x, masks)

    # print
    # img_grid=utils.make_grid(y_hat)
    # img = T.ToPILImage()(img_grid)
    # img.show()

    # img_grid=utils.make_grid(original)
    # img = T.ToPILImage()(img_grid)
    # img.show()

    # the plan is to do multiple experiments for each model we can train on all the datasets
    # three models, small, medium, big
    # one dense network