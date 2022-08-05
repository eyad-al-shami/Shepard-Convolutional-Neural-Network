import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import PreprocessedImageInpaintingDataset
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
from torchvision import utils
from torchvision import transforms as T
from configs import training_configs, experiments_config, data_set_config, seed
import argparse
import os
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from model import ShepardNet

# fix the seed for reproducibility
seed = seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def train(args):
    start = time.time()
    batch_size = training_configs['batch_size']()
    print("========================================================")
    print(f"Based on the GPU memory, the batch size is {batch_size}")
    print(f"Logging {'with' if args.wandb_log else 'without'} wandb.")
    print(f"All models are going to be trained for {training_configs['epochs']} epochs.")
    print("========================================================")

    for data_path in data_set_config['paths']:
        print(f"Trainig on {os.path.basename(data_path)} training dataset")
        # Train, Test splits
        dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])
        dataset_size = len(dataset)
        split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))
        train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))
        # Train, Validation splits
        trainset_size = len(train_set)
        split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))
        train_set, validation_set = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))
        # printing the sizes of the datasets
        print(f"\tTraining dataset contains {len(train_set)} examples\n\tValidation dataset conttains {len(validation_set)}\n\tTest dataset contains {len(test_set)}")

        # The beginning of the training
        for experiment in experiments_config['experiments']:
            print(f"\n>>>>>>>>   Model being trained is {experiment['name']}   <<<<<<<<\n")
            train_dataloader = DataLoader(train_set, batch_size=experiment['batch_size'],
                            shuffle=True, num_workers=10, persistent_workers=True)

            validation_dataloader = DataLoader(validation_set, batch_size=experiment['batch_size'],
                                shuffle=False, num_workers=2, persistent_workers=True)
            layers = experiment['layers']
            net = ShepardNet(layers, training_configs['LR'])

            if (args.wandb_log):
                wandb_logger = WandbLogger(project="ShCNN", name=experiment['name'], log_model="all", save_dir=args.log_path)
                wandb_logger.experiment.config.update({'layers': layers}, allow_val_change=True)
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True, logger=wandb_logger)
            else:
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=training_configs['epochs'], deterministic=True, default_root_dir=args.log_path)

            model_trainig_time = time.time()
            trainer.fit(net, train_dataloader, validation_dataloader)
            print(f"---------- Model took {(time.time() - model_trainig_time) / 60.0} minutes. ----------")

            if (args.wandb_log):
                wandb_logger.experiment.finish()

    print("========================================================")
    print(f"Training took {(time.time() - start) / 60.0} minutes.")
    print("========================================================\n")

def infer(args, batch_size=1):
    '''
    Inferring the model on a test dataset
    '''

    if (not args.model_path or not args.data_path):
        raise Exception("model_path and data_path must be specified")

    # recreate the datasets
    data_path = args.data_path
    dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])
    dataset_size = len(dataset)
    split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))
    train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))
    trainset_size = len(train_set)
    split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))
    train_set, _ = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))
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

    model_path = r"C:\Users\eyad\Documents\CODE\Training output\lightning_logs\version_0\checkpoints\epoch=4-step=1500.ckpt"
    model_path = args.model_path
    net = ShepardNet.load_from_checkpoint(model_path)
    net.eval()

    # predict with the model
    y_hat, masks = net(x, masks)

    # print
    img_grid=utils.make_grid(y_hat)
    img = T.ToPILImage()(img_grid)
    img.save('/content/outputs/prediction.png')

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

    