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
from pytorch_lightning.callbacks import ModelCheckpoint

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

        print(len(self.layers))

        self.modules_list = nn.ModuleList()
        for i, (input_layer, output_layer) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            if (output_layer.layer_type == 'conv'):
                self.modules_list.append(nn.Conv2d(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
            else:
                self.modules_list.append(ShConv(input_layer.kernels_num, output_layer.kernels_num, output_layer.kernel_size, stride=output_layer.stride, padding=output_layer.padding))
            self.modules_list.append(nn.ReLU())
            self.modules_list.append(nn.BatchNorm2d(output_layer.kernels_num))

        # saving the hyperparameters (for wandb)
        self.save_hyperparameters()
        self.loss_function = torch.nn.MSELoss()

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
        loss = self.loss_function(x, original)
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
        test_loss = self.loss_function(x, original)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        original, x, masks =  batch
        for layer in self.modules_list:
          if isinstance(layer, ShConv):
              x, masks = layer(masks, x)
          else:
              x = layer(x)
        val_loss = self.loss_function(x, original)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        return optimizer


def train(args):
    
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")
    

    start = time.time()

    batch_size = training_configs['batch_size']()
    print("========================================================")
    print(f"Based on the GPU memory, the batch size is {batch_size}")
    print(f"Logging {'with' if args.wandb_log else 'without'} wandb.")
    print(f"All models are going to be trained for {training_configs['epochs']} epochs.")
    print("========================================================")



    for data_path in data_set_config['paths']:
        print(f"Trainig on {os.path.basename(data_path)} training dataset")

        dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])

        dataset_size = len(dataset)

        split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))

        train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))

        trainset_size = len(train_set)

        split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))

        train_set, validation_set = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))

        print(f"\tTraining dataset contains {len(train_set)} examples\n\tValidation dataset conttains {len(validation_set)}\n\tTest dataset contains {len(test_set)}")

        # training _____________________________________________________________________________________

        for experiment in experiments_config['experiments']:
            print(f"\n    >>>>>>>>   Model being trained is {experiment['name']}   <<<<<<<<\n")

            train_dataloader = DataLoader(train_set, batch_size=experiment['batch_size'],
                            shuffle=True, num_workers=10, persistent_workers=True)

            validation_dataloader = DataLoader(validation_set, batch_size=experiment['batch_size'],
                                shuffle=False, num_workers=2, persistent_workers=True)

            layers = experiment['layers']

            net = ShepardNet(layers, training_configs['LR'])

            if (args.wandb_log):
                wandb_logger = WandbLogger(project="ShCNN", name=experiment['name'], log_model="all", save_dir=args.log_path)
                wandb_logger.experiment.config.update({'layers': layers}, allow_val_change=True)
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True, logger=wandb_logger, callbacks=[checkpoint_callback])
            else:
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True, default_root_dir=args.log_path, callbacks=[checkpoint_callback])

            model_trainig_time = time.time()
            trainer.fit(net, train_dataloader, validation_dataloader)
            print(f"    ---------- Model took {(time.time() - model_trainig_time) / 60.0} minutes. ----------")
            if (args.wandb_log):
                wandb_logger.experiment.finish()

        # training _____________________________________________________________________________________

    print("========================================================")
    print(f"Training took {(time.time() - start) / 60.0} minutes.")
    print("========================================================")

def infer(args):
  print(args)

  if (not args.model_path or not args.data_path):
    raise Exception("model_path and data_path must be specified")

  # data_path = r"C:\Users\eyad\Pictures\Images Datasets\1_cutout_large_50px"
  data_path = args.data_path
  batch_size = 1


  dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])

  dataset_size = len(dataset)

  split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))

  train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))

  trainset_size = len(train_set)

  split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))

  train_set, validation_set = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))

  print(f"\tTraining dataset contains {len(train_set)} examples\n\tValidation dataset conttains {len(validation_set)}\n\tTest dataset contains {len(test_set)}")

  train_dataloader = DataLoader(train_set, batch_size=batch_size,
                      shuffle=True, num_workers=8)

  validation_dataloader = DataLoader(validation_set, batch_size=batch_size,
                      shuffle=False, num_workers=4)

  test_dataloader = DataLoader(test_set, batch_size=batch_size,
                      shuffle=False, num_workers=2)

  dataloader_iter = iter(test_dataloader)
  original, x, masks = next(dataloader_iter)

  img_grid=utils.make_grid(original)
  img = T.ToPILImage()(img_grid)
  img.save('original.png')

  img_grid=utils.make_grid(x)
  img = T.ToPILImage()(img_grid)
  img.save('corrupted.png')

  img_grid=utils.make_grid(masks)
  img = T.ToPILImage()(img_grid)
  img.save('mask.png')

  # model_path = r"C:\Users\eyad\Documents\CODE\Training output\lightning_logs\version_0\checkpoints\epoch=4-step=1500.ckpt"
  model_path = args.model_path
  net = ShepardNet.load_from_checkpoint(model_path)
  net.eval()

  # predict with the model
  y_hat, masks = net(x, masks)

  # print
  img_grid=utils.make_grid(y_hat)
  img = T.ToPILImage()(img_grid)
  img.save('prediction.png')

def main(args):
  if (args.train):
    train(args)
  if (args.infer):
    infer(args)
  else:
    layers = experiments_config['experiments'][0]['layers']
    net = ShepardNet(layers, training_configs['LR'])
    print(net)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', help="boolean value to indicate using wandb log or not.", action="store_true", default=False)
    parser.add_argument('--train', help="boolean value to indicate the training phase.", action="store_true", default=False)
    parser.add_argument('--log-path', help="Path to save logs", type=str, default="")

    parser.add_argument('--infer', help="boolean value to indicate the infering phase.", action="store_true", default=False)
    parser.add_argument('--data-path', help="path for the data.", type=str, default="")
    parser.add_argument('--model-path', help="path for saved model.", type=str, default="")

    
    args = parser.parse_args()
    main(args)


    # DUMMY Training

    