import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import PreprocessedImageInpaintingDataset, ImageInpaintingDataset
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
from matplotlib import pyplot as plt
from pathlib import Path
from transforms import CutOutRectangles, RandomText, ToTensor

# fix the seed for reproducibility
seed = seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_corruption_transform(transform):
    if transform == "text_corruption":
        return RandomText(text_size=18)
    elif transform == "cutout_corruption":
        return CutOutRectangles(num_rectangles=1, max_h_size=50, max_w_size=50)
    else:
        raise Exception("transform must be specified")


def get_data_sets(data_path, data_extension=["png", "jpg"], preprocessed_data=True, transform=None):
    '''
        Read the dataset from the given path and split it into train, validation and test sets.
    '''
    if (preprocessed_data):
        dataset = PreprocessedImageInpaintingDataset(data_path, extensions=data_set_config['extensions'])
    else:
        corruption = get_corruption_transform(transform)
        dataset = ImageInpaintingDataset(data_path, extensions=data_extension, transform=corruption)

    dataset_size = len(dataset)
    split = int(np.floor(data_set_config['splits']['TEST_SPLIT'] * dataset_size))
    train_set, test_set = random_split(dataset, [dataset_size - split, split], generator=torch.Generator().manual_seed(seed))
    # Train, Validation splits
    trainset_size = len(train_set)
    split = int(np.floor(data_set_config['splits']['VALIDATION_SPLIT'] * trainset_size))
    train_set, validation_set = random_split(train_set, [trainset_size - split, split], generator=torch.Generator().manual_seed(seed))
    return train_set, validation_set, test_set

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
        train_set, validation_set, test_set = get_data_sets(data_path, data_extension=data_set_config['extensions'], preprocessed_data=data_set_config['preprocessed_data'], transform=data_set_config['transform'])

        print(f"\tTraining dataset contains {len(train_set)} examples\n\tValidation dataset conttains {len(validation_set)}\n\tTest dataset contains {len(test_set)}")

        # The beginning of the training
        for experiment in experiments_config['experiments']:
            print(f"\n>>>>>>>>   Model being trained is {experiment['name']}   <<<<<<<<\n")
            train_dataloader = DataLoader(train_set, batch_size=experiment['batch_size'],
                            shuffle=True, num_workers=8, persistent_workers=True)

            validation_dataloader = DataLoader(validation_set, batch_size=experiment['batch_size'],
                                shuffle=False, num_workers=4, persistent_workers=True)
            layers = experiment['layers']
            net = ShepardNet(layers, training_configs['LR'])
            
            epochs =  args.epochs or training_configs['epochs']
            print(f"Training for {epochs} epcohs...")
            
            log_path = args.log_path
            checkpoint_callback = ModelCheckpoint(monitor="val_loss")

            if (args.wandb_log):
                wandb_logger = WandbLogger(project="ShCNN", name=experiment['name']+os.path.basename(data_path), log_model="all", save_dir=log_path)
                wandb_logger.experiment.config.update({'layers': layers}, allow_val_change=True)
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=epochs, deterministic=True, logger=wandb_logger, gradient_clip_val=1.0, callbacks=[checkpoint_callback])
            else:
                trainer = pl.Trainer(accelerator=training_configs['accelerator'], max_epochs=epochs, deterministic=True, default_root_dir=log_path, gradient_clip_val=1.0, callbacks=[checkpoint_callback])

            model_trainig_time = time.time()
            trainer.fit(net, train_dataloader, validation_dataloader)
            print(f"---------- Model took {(time.time() - model_trainig_time) / 60.0} minutes. ----------")

            if (args.wandb_log):
                wandb_logger.experiment.finish()

    print("========================================================")
    print(f"Training took {(time.time() - start) / 60.0} minutes.")
    print("========================================================\n")

def infer_cmd(args, batch_size: int = 1, transform: str = "text_corruption"):
    '''
        Inferring the model on a test dataset
        batch_size: the batch size to use for inference
        transform: the transform to use for inference, can be one of the following:
            - text_corruption: text corruption
            - cutout_corruption: rectangular cutout
    '''

    if (not args.model_path or not args.data_path):
        raise Exception("model_path and data_path must be specified")

    # recreate the datasets
    data_path = args.data_path
    train_set, validation_set, test_set = get_data_sets(data_path)
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    dataloader_iter = iter(test_dataloader)
    original, x, masks = next(dataloader_iter)

    img_grid=utils.make_grid(x)
    img = T.ToPILImage()(img_grid)
    img.save('corrupted.png')

    img_grid=utils.make_grid(masks)
    img = T.ToPILImage()(img_grid)
    img.save('mask.png')

    model_path = args.model_path
    net = ShepardNet.load_from_checkpoint(model_path)
    net.eval()

    # predict with the model
    y_hat, masks = net(x, masks)

    img_grid=utils.make_grid(y_hat)
    img = T.ToPILImage()(img_grid)
    img.save('prediction.png')
    
    for i in range(8):
        img_grid=utils.make_grid(masks[0,i])
        img = T.ToPILImage()(img_grid)
        img.save(f'masks_{i}.png')

def infer(model_path, data_path:str, batch_size: int=1, data_extension=["png", "jpg"], preprocessed_data:bool=True, transform:str = "text_corruption", padding:int=1, show_masks:bool=False):
    '''
        Inferring the model on a test dataset
        data_path: the path to the test dataset
        batch_size: the batch size to use for inference
        data_extension: the extension of the test dataset images
        preprocessed_data: whether the test dataset is preprocessed
        transform: the transform to use for inference, can be one of the following:
            - text_corruption: text corruption
            - cutout_corruption: rectangular cutout
        padding: the padding to use for the showing the images

    '''
    assert not model_path or not data_path, "model_path and data_path must be specified."
    assert not preprocessed_data and not transform, "you can either use preprocessed data or transform/corrupt an original data, you have to specify one of them."

    p = Path(model_path)
    model_name = p.parts[1]

    # recreate the datasets
    train_set, validation_set, test_set = get_data_sets(data_path, data_extension=data_extension, preprocessed_data=preprocessed_data, transform=transform)
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    dataloader_iter = iter(test_dataloader)
    original, corrupted, masks = next(dataloader_iter)

    net = ShepardNet.load_from_checkpoint(model_path)
    net.eval()

    # predict with the model
    y_hat, masks = net(corrupted, masks)

    pred_grid=utils.make_grid(y_hat, nrow=4, padding=padding,).permute(1, 2, 0)
    corrupted_grid=utils.make_grid(corrupted, nrow=4, padding=padding,).permute(1, 2, 0)
    original_grid=utils.make_grid(original,nrow=4, padding=padding,).permute(1, 2, 0)


    fig, ax = plt.subplots(1, 3, figsize=(30, 7))
    fig.suptitle(model_name, fontsize=18)
    ax[0].imshow(original_grid)
    ax[0].set_title('Original', fontsize=18)
    ax[1].imshow(corrupted_grid)
    ax[1].set_title('Corrupted', fontsize=18)
    ax[2].imshow(pred_grid)
    ax[2].set_title('Prediction', fontsize=18)
    fig.tight_layout()

    if (show_masks):
        fig, ax = plt.subplots(2, 8, figsize=(30, 7))
        fig.suptitle("Final Masks", fontsize=18)
        for j in range(batch_size):
            for i in range(8):
                img_grid=utils.make_grid(masks[j,i]).permute(1, 2, 0)
                # img = T.ToPILImage()(img_grid)
                ax[j, i].imshow(img_grid)
                ax[j, i].set_title(f'{i+1}{"th" if i+1 > 3 else "nd" if i+1 == 2 else "st" if i+1 == 1 else "rd"} mask', fontsize=18)
        fig.tight_layout()

def main(args):
  if (args.train):
    train(args)
  if (args.infer):
    infer_cmd(args)
  else:
    layers = experiments_config['experiments'][0]['layers']
    net = ShepardNet(layers, training_configs['LR'])
    print(net)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-log', help="boolean value to indicate using wandb log or not.", action="store_true", default=False)
    parser.add_argument('--train', help="boolean value to indicate the training phase.", action="store_true", default=False)
    parser.add_argument('--log-path', help="Path to save logs", type=str, default="")
    parser.add_argument('--preprocessed-data', help="A flag indicates if the data is already preprocessed/corrupted or not.", action="store_true", default=False)
    parser.add_argument('--preprocessed-data', help="A flag indicates if the data is already preprocessed/corrupted or not.", action="store_true", default=False)
    parser.add_argument('--epochs', help="Number of epochs to overwrite the default one.", type=int, default=0)
    

    parser.add_argument('--infer', help="boolean value to indicate the infering phase.", action="store_true", default=False)
    parser.add_argument('--data-path', help="path for the data.", type=str, default="")
    parser.add_argument('--model-path', help="path for saved model.", type=str, default="")

    
    args = parser.parse_args()
    main(args)

    

    