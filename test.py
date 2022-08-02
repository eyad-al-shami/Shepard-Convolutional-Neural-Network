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
import os
import time

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
                print('shconv layer')
                x, masks = layer(masks, x)
            else:
                print('conv layer')
                x = layer(x)
                print(x.shape)
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


if __name__ == "__main__":

    print(training_configs['batch_size']())
    # print(output_features_map.shape)
    # print(output_mask.shape)
    

