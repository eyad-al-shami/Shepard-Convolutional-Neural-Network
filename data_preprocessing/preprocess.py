from re import X
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.font_manager
import random
from PIL import ImageFont
from PIL import ImageDraw
from glob import glob
import os


seed = 877
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class CutOutRectangles(object):
    """Cut out randomly rectangles from the image.

    Args:
        num_rectangles (int): Number of rectangles to cut out.
        max_h_size (int): Maximum height of the cut out rectangle.
        max_w_size (int): Maximum width of the cut out rectangle.
    """

    def __init__(
        self,
        root_path,
        num_rectangles: int = 1,
        max_h_size: int = 40,
        max_w_size: int = 40
        ):
        print("inside init of CutOutRectangles")
        self.num_rectangles = num_rectangles
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.original = os.path.join(root_path, f'original_{max_h_size}px')
        self.corrupted = os.path.join(root_path, f'corrupted_{max_h_size}px')
        self.mask = os.path.join(root_path, f'mask_{max_h_size}px')

        for p in [self.original, self.corrupted, self.mask]:
            if not os.path.exists(p):
                os.makedirs(p)



    def __call__(self, original_path : str):
        with Image.open(original_path) as original:
            image = np.array(original)
            mask = np.ones_like(image) * 255.
            h, w = image.shape[:2]

            # create the corners of the cutout rectangle

            for i in range(self.num_rectangles):
                y = torch.randint(0, h, (1, )).item()
                x = torch.randint(0, w, (1, )).item()

                y1 = np.clip(y - self.max_h_size // 2, 0, h)
                y2 = np.clip(y1 + self.max_h_size, 0, h)
                x1 = np.clip(x - self.max_w_size // 2, 0, w)
                x2 = np.clip(x1 + self.max_w_size, 0, w)

                # set the values in the  recagle in the image to 0
                image[y1:y2, x1:x2, :] = 0.
                # using an RGB mode for the input image in acceptable because we want mask for each input channel
                mask[y1:y2, x1:x2, :] = 0.
            # Image.fromarray(image.astype(np.uint8)).save("image-intermediate.png")
            # Image.fromarray(mask.astype(np.uint8)).save("mask-intermediate.png")
            # return {'original': original, 'corrupted': Image.fromarray(image.astype(np.uint8)), 'mask': Image.fromarray(mask.astype(np.uint8))}

            original.save(os.path.join(self.original, os.path.basename(original_path)))

            Image.fromarray(image.astype(np.uint8)).save(os.path.join(self.corrupted, os.path.basename(original_path)))

            Image.fromarray(mask.astype(np.uint8)).save(os.path.join(self.mask, os.path.basename(original_path)))


class RandomText(object):
    """Add text on image .

    Args:
        text (str): Text to add.
        text_size (int): Size of the text.
        font (str): Font to use.
    """

    def __init__(self, text_size: int, root_path, font=None):
        self.text_size = text_size
        self.font = font
        if font is None:
            self.fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        
        # mit words list file, the max word length is 22
        # https://www.mit.edu/~ecprice/wordlist.10000
        self.max_word_length = 22
        self.max_word_length_text_size_rel = self.max_word_length * (self.text_size//2)
        with open('mit-words.txt', 'r') as f:
            self.words = f.read().splitlines()

    def __call__(self, original: Image):

        try:
          # font_name = random.choice(self.fonts)
          font = ImageFont.truetype("arial.ttf", self.text_size)
        except:
          font_name = random.choice(self.fonts)
          font = ImageFont.truetype(font_name, self.text_size)
          
        image = original.copy()
        image_draw = ImageDraw.Draw(image)
        mask = Image.new(mode="RGB", size=image.size, color = 'white')
        mask_draw = ImageDraw.Draw(mask)

        # in PIL, size returns (width, height)
        num_words = image.size[0] // self.text_size
        words = " ".join(np.random.choice(self.words, num_words))

        randomness_range = 50

        # while we are drawing the text in coordiate smaller than the image's hight continue adding text
        slack = np.random.choice(randomness_range, 1)[0]
        # x = np.random.choice(randomness_range, 1)[0]
        x = 0
        y = np.random.choice(randomness_range, 1)[0]
        
        height = image.size[1]
        slack_range = np.arange(self.text_size,randomness_range)

        while(y <= height):
            image_draw.text((x, y + slack), words, (0, 0, 0), font=font)
            mask_draw.text((x, y + slack), words, (0, 0, 0), font=font)
            slack = np.random.choice(slack_range, 1)[0]
            x = np.random.choice(randomness_range, 1)[0]
            y = y + slack
            words = " ".join(np.random.choice(self.words, num_words))
            # font_name = random.choice(self.fonts)

        self.original = os.path.join(self.root_path, 'original')
        self.corrupted = os.path.join(self.root_path, 'corrupted')
        self.mask = os.path.join(self.root_path, 'mask')

        for p in [self.original, self.corrupted, self.mask]:
            if not os.path.exists(p):
                os.makedirs(p)


def read_images(root_dir, extensions=['jpg'], min_size=None, transform=None, nested=False):
    """
    Args:
        root_dir (string): Directory with all the images.
        extensions (list of strings, optional): List of allowed image extensions.
        min_size (int, optional): Minimum size of the image.
        transform (callable, optional): Optional transform to be applied on a sample.
        nested (bool, optional): if True, images are in a nested directory structure (only one level of nesting).
    """
    root_dir = root_dir
    transform = transform
    # use glob to get a list of all images in the root_dir
    # check the type of extensions, if it is a list, use it, otherwise make it a list
    if not isinstance(extensions, list) and isinstance(extensions, str):
        extensions = extensions.split(',')
    images = []
    search_pattern = '*.{}' if not nested else '**/*.{}'
    for extension in extensions:
        images.extend(glob(os.path.join(root_dir, search_pattern.format(extension))))
    if (min_size is not None):
        print("Filtering images based on the specified min_size...")
        for img_p in images:
            with Image.open(img_p) as img:
                (width, height) = img.size
                if (width < min_size or height < min_size):
                    images.remove(img_p)
        print("Done.")
    return images




# randTextFun = RandomText(20)

from multiprocessing import Pool


if __name__ == "__main__":


    datasets = [
        {
            'name': "1_cutout_small_20px",
            'parameters': {
                'cutouts': 1,
                'max_size': 20
            }
        },
        {
            'name': "1_cutout_large_50px",
            'parameters': {
                'cutouts': 1,
                'max_size': 50
            }
        },
        {
            'name': "2_cutouts_small_20px",
            'parameters': {
                'cutouts': 2,
                'max_size': 20
            }
        },
        {
            'name': "2_cutouts_large_50px",
            'parameters': {
                'cutouts': 2,
                'max_size': 50
            }
        }
    ]

    original_data_path = r"C:\Users\eyad\Pictures\Images Datasets\Filcker Faces thumbnails 128x128"
    base_directory = os.path.dirname(original_data_path)

    images = read_images(original_data_path, extensions=["png"], nested=True)

    for dataset in datasets:
        dataset_path = os.path.join(base_directory, dataset['name'])
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    
        cutOutFun = CutOutRectangles(dataset_path, num_rectangles=dataset['parameters']['cutouts'], max_h_size=dataset['parameters']['max_size'], max_w_size=dataset['parameters']['max_size'])

        with Pool(5) as p:
            p.map(cutOutFun, images)
        print(f"Done {dataset['name']}")




