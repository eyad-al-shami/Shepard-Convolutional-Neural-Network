
from utils import LayersHyperParameters
import torch

seed = 877


def get_batch_size():
    try:
        _, total = torch.cuda.mem_get_info(0)
        total = round(total/1024**3,1)
        if (total >= 40):
            return 100
        elif (total >= 16):
            return 32
        else:
            return 16
    except Exception as e:
        print(e)
        return 16


data_set_config = {
    "splits": {
        "TEST_SPLIT": 0.05,
        "VALIDATION_SPLIT": 0.1
    },
    "extensions": ["png"],
    "paths": [
        r"C:\Users\eyad\Pictures\Images Datasets\1_cutout_large_50px",
        # r"C:\Users\eyad\Pictures\Images Datasets\2_cutouts_small_20px"
    ]
}

training_configs = {
    "epochs": 4,
    "LR": 1e-3,
    "batch_size": get_batch_size,
    "accelerator": "gpu"
}

experiments_config = {
    "project": "ShCNN",
    "sizes": {
        "small": 5,
        "medium": 7
    },
    "experiments": [
        # {
        #     "name": "large_b_6sl",
        #     "layers": [
        #         LayersHyperParameters("shepard", 32, 5),
        #         LayersHyperParameters("shepard", 32, 5),
        #         LayersHyperParameters("shepard", 64, 3),
        #         LayersHyperParameters("shepard", 64, 3),
        #         LayersHyperParameters("shepard", 128, 3),
        #         LayersHyperParameters("shepard", 128, 3),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 256, 3),
        #         LayersHyperParameters("conv", 256, 1),
        #         LayersHyperParameters("conv", 3, 3),
        #     ]
        # },
        # {
        #     "name": "medium_b_4sl",
        #     "layers": [
        #         LayersHyperParameters("shepard", 32, 5),
        #         LayersHyperParameters("shepard", 64, 5),
        #         LayersHyperParameters("shepard", 64, 3),
        #         LayersHyperParameters("shepard", 128, 3),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 256, 1),
        #         LayersHyperParameters("conv", 3, 3),
        #     ]
        # },
        # {
        #     "name": "small_b_2sl",
        #     "layers": [
        #         LayersHyperParameters("shepard", 32, 5),
        #         LayersHyperParameters("shepard", 64, 5),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 128, 1),
        #         LayersHyperParameters("conv", 3, 3),
        #     ]
        # },
        {
            "name": "small_b_2sl",
            "batch_size": 64,
            "layers": [
                LayersHyperParameters("shepard", 32, 5),
                LayersHyperParameters("shepard", 64, 5),
                LayersHyperParameters("conv", 128, 3),
                LayersHyperParameters("conv", 128, 1),
                LayersHyperParameters("conv", 3, 3),
            ]
        },
    ]
}