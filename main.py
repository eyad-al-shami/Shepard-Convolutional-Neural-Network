import torch
import torch.nn as nn
import torch.nn.functional as F
from ShConv import ShConv
from utils import LayersHyperParameters
import pytorch_lightning as pl


class ShepardNet(pl.LightningModule):
    def __init__(self, layers):
        super(ShepardNet, self).__init__()
        self.layers = layers
        the_input_layer = LayersHyperParameters(self.layers[0].layer_type, 3, self.layers[0].kernel_size)
        self.layers = [the_input_layer, *self.layers]
        self.modules_list = nn.ModuleList()
        for i, (input_layer, output_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            if (output_layer.layer_type == 'conv'):
                self.modules_list.append(nn.Conv2d(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
                self.modules_list.append(nn.ReLU())
                if (i != len(self.layers) - 1):
                    self.modules_list.append(nn.BatchNorm2d(output_layer.kernels_num))
            else:
                self.modules_list.append(ShConv(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
                self.modules_list.append(nn.ReLU())
                if (i != len(self.layers) - 1):
                    self.modules_list.append(nn.BatchNorm2d(output_layer.kernels_num))

    # def forward(self, masks, x):
    #     # seen_first_sh_layer = False
    #     for layer in self.modules_list:
    #         # if not seen_first_sh_layer and isinstance(layer, ShConv):
    #         #     seen_first_sh_layer = True
    #         #     x, intermediate_masks = layer(masks, x)
    #         # elif seen_first_sh_layer and isinstance(layer, ShConv):
    #         #     x, intermediate_masks = layer(intermediate_masks, x)
    #         if isinstance(layer, ShConv):
    #             x, masks = layer(masks, x)
    #         else:
    #             x = layer(x)
    #     return x, masks

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        original, x, masks = batch['original'], batch['corrupted'], batch['mask']
        for layer in self.modules_list:
            if isinstance(layer, ShConv):
                x, masks = layer(masks, x)
            else:
                x = layer(x)
        loss = F.mse_loss(original, x)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        original, x, masks = batch['original'], batch['corrupted'], batch['mask']
        for layer in self.modules_list:
            if isinstance(layer, ShConv):
                x, masks = layer(masks, x)
            else:
                x = layer(x)
        test_loss = F.mse_loss(original, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        original, x, masks = batch['original'], batch['corrupted'], batch['mask']
        for layer in self.modules_list:
            if isinstance(layer, ShConv):
                x, masks = layer(masks, x)
            else:
                x = layer(x)
        test_loss = F.mse_loss(original, x)
        self.log("val_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    layers = [
        LayersHyperParameters('shepard', 8, 7),
        LayersHyperParameters('shepard', 8, 5),
        LayersHyperParameters('conv', 10, 3),
        LayersHyperParameters('conv', 25, 3),
        LayersHyperParameters('conv', 3, 3),
    ]
    net = ShepardNet(layers)

    example_input = torch.randn(1, 3, 100, 100)
    example_masks = torch.randn(1, 3, 100, 100)
    example_output, intermediate_masks = net(example_masks, example_input)
    print(intermediate_masks.shape)
    print(example_output.shape)