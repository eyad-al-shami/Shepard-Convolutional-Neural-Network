from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from glob import glob
import os
from PIL import Image

from transforms import CutOutRectangles, RandomText, ToTensor

class ImageInpaintingDataset(Dataset):

    def __init__(self, root_dir, extensions=['jpg'], min_size=None, transform=None, nested=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            extensions (list of strings, optional): List of allowed image extensions.
            min_size (int, optional): Minimum size of the image.
            transform (callable, optional): Optional transform to be applied on a sample.
            nested (bool, optional): if True, images are in a nested directory structure (only one level of nesting).
        """
        self.root_dir = root_dir
        self.transform = transform
        # use glob to get a list of all images in the root_dir
        # check the type of extensions, if it is a list, use it, otherwise make it a list
        if not isinstance(extensions, list) and isinstance(extensions, str):
            extensions = extensions.split(',')
        self.images = []
        search_pattern = '*.{}' if not nested else '**/*.{}'
        for extension in extensions:
            self.images.extend(glob(os.path.join(root_dir, search_pattern.format(extension))))
        if (min_size is not None):
            print("Filtering images based on the specified min_size...")
            for img_p in self.images:
              with Image.open(img_p) as img:
                  (width, height) = img.size
                  if (width < min_size or height < min_size):
                      self.images.remove(img_p)
            print("Done.")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name)
        sample = {'original': image}
        if self.transform:
            sample = self.transform(image)
        return sample


if __name__ == '__main__':
    print('main')
    import numpy as np
    # transform = T.ToPILImage()
    inpaintingDataset = ImageInpaintingDataset(root_dir='./images', transform=T.Compose([
                                            #    CutOutRectangles(num_rectangles=1),
                                               RandomText(text_size=25),
                                               ToTensor()
                                           ]))
    dataloader = DataLoader(inpaintingDataset, batch_size=2,
                        shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['mask'].size(),
            sample_batched['corrupted'].size())