from dataclasses import dataclass
from typing import Union
@dataclass
class LayersHyperParameters():
    layer_type: str = 'conv'
    kernels_num: int = 128
    kernel_size: int = 3
    stride: int = 1
    padding: Union[str, int] = "same"