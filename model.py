from ShConv import ShConv
import pytorch_lightning as pl
from utils import LayersHyperParameters
import torch
import torch.nn as nn



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
            self.modules_list.append(nn.BatchNorm2d(output_layer.kernels_num))
            self.modules_list.append(nn.ReLU())
            
        self.loss_function = torch.nn.MSELoss()
        # saving the hyperparameters (for wandb)
        self.save_hyperparameters()

    def forward(self, x, masks):
        for layer in self.modules_list:
            if isinstance(layer, ShConv):
                print(layer)
                x, masks = layer(masks, x)
            else:
                print(layer)
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
        loss = self.loss_function(x, original)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        test_loss = self.loss_function(x, original)
        self.log("test_loss", test_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        val_loss = self.loss_function(x, original)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        return optimizer