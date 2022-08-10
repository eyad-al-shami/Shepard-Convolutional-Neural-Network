
from utils import LayersHyperParameters
import torch

# The random seed for the experiment
seed = 877

def get_batch_size():
    '''
        Returns a batch size that is appropriate for the current GPU
    '''
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
        r"../Flicker_faces_128/random_text_20px",
        # r"../Flicker_faces_128/1_cutout_large_50px"
    ]
}

training_configs = {
    "epochs": 4,
    "LR": 1e-2,
    "batch_size": get_batch_size,
    "accelerator": "gpu"
}

experiments_config = {
    "project": "ShCNN",
    "experiments": [
        # {
        #     "name": "small_b_3sl",
        #     "batch_size": 128,
        #     "layers": [
        #         LayersHyperParameters("shepard", 8, 7),
        #         LayersHyperParameters("shepard", 16, 5),
        #         LayersHyperParameters("shepard", 32, 5),
        #         LayersHyperParameters("conv", 64, 5),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 128, 3),
        #         LayersHyperParameters("conv", 3, 3),
        #     ]
        # },
        {
            "name": "base_model",
            "batch_size": 250,
            "layers": [
                LayersHyperParameters("shepard", 8, 4),
                LayersHyperParameters("shepard", 8, 4),
                LayersHyperParameters("conv", 128, 9),
                LayersHyperParameters("conv", 128, 1),
                LayersHyperParameters("conv", 3, 8),
            ]
        },
        # {
        #     "name": "test_model",
        #     "batch_size": 250,
        #     "layers": [
        #         LayersHyperParameters("shepard", 8, 5),
        #         LayersHyperParameters("shepard", 8, 5),
        #         LayersHyperParameters("conv", 128, 9),
        #         LayersHyperParameters("conv", 128, 1),
        #         LayersHyperParameters("conv", 3, 7),
        #     ]
        # },
    ]
}