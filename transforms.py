from re import X
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.font_manager
import random
from PIL import ImageFont
from PIL import ImageDraw 

class CutOutRectangles(object):
    """Cut out randomly rectangles from the image.

    Args:
        num_rectangles (int): Number of rectangles to cut out.
        max_h_size (int): Maximum height of the cut out rectangle.
        max_w_size (int): Maximum width of the cut out rectangle.
    """

    def __init__(
        self,
        num_rectangles: int = 1,
        max_h_size: int = 40,
        max_w_size: int = 40
        ):
        print("inside init of CutOutRectangles")
        self.num_rectangles = num_rectangles
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, original : Image):
        
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
        return {'original': original, 'corrupted': Image.fromarray(image.astype(np.uint8)), 'mask': Image.fromarray(mask.astype(np.uint8))}

class RandomText(object):
    """Add text on image .

    Args:
        text (str): Text to add.
        text_size (int): Size of the text.
        font (str): Font to use.
    """

    def __init__(self, text_size: int, font=None):
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

        return {'original': original, 'corrupted': image, 'mask': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, sample):
        Original, corrupted, mask = sample['original'], sample['corrupted'], sample['mask']
        return {'original': self.to_tensor(Original), 'corrupted': self.to_tensor(corrupted), 'mask': self.to_tensor(mask)}

