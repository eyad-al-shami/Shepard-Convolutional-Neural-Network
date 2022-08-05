from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from glob import glob
import os
from PIL import Image
from torchvision import utils

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


class PreprocessedImageInpaintingDataset(Dataset):

    def __init__(self, root_dir, extensions=['jpg'], transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            extensions (list of strings, optional): List of allowed image extensions.
            min_size (int, optional): Minimum size of the image.
            transform (callable, optional): Optional transform to be applied on a sample.
            nested (bool, optional): if True, images are in a nested directory structure (only one level of nesting).
        """

        self.transforms = T.Compose([
            T.ToTensor(),
        ])

        self.data = {
            "original": {
                "root_path": "",
                "images_paths": []
            },
            "corrupted": {
                "root_path": "",
                "images_paths": []
            },
            "mask": {
                "root_path": "",
                "images_paths": []
            }
        }

        search_pattern = '*.{}'

        self.root_dir = root_dir
        # use glob to get a list of all images in the root_dir
        # check the type of extensions, if it is a list, use it, otherwise make it a list
        if not isinstance(extensions, list) and isinstance(extensions, str):
            extensions = extensions.split(',')
        
        self.images_sources = [os.path.join(self.root_dir, name) for name in os.listdir(self.root_dir)]

        self.data["original"]["root_path"] = next((x for x in self.images_sources if 'original' in x), None)
        self.data["corrupted"]["root_path"] = next((x for x in self.images_sources if 'corrupted' in x), None)
        self.data["mask"]["root_path"] = next((x for x in self.images_sources if 'mask' in x), None)

        for image_source in self.data:
            for extension in extensions:
                self.data[image_source]["images_paths"].extend(glob(os.path.join(self.data[image_source]["root_path"], search_pattern.format(extension))))

    def __len__(self):
        return len(self.data['corrupted']['images_paths'])

    def __getitem__(self, idx):
        original_path = self.data["original"]["images_paths"][idx]
        corrupted_path = self.data["corrupted"]["images_paths"][idx]
        mask_path = self.data["mask"]["images_paths"][idx]

        original = Image.open(original_path)
        corrupted = Image.open(corrupted_path)
        mask = Image.open(mask_path)

        return self.transforms(original), self.transforms(corrupted), self.transforms(mask)
        



if __name__ == '__main__':
    import numpy as np
    
    # THIS CODE IS FOR THE DATASET THAT USES THE ORIGINAL IMAGES AND TRANSFORMS THEM ON THE FLY.

    # transform = T.ToPILImage()
    # inpaintingDataset = ImageInpaintingDataset(root_dir='./images', transform=T.Compose([
    #                                         #    CutOutRectangles(num_rectangles=1),
    #                                            RandomText(text_size=25),
    #                                            ToTensor()
    #                                        ]))
    # dataloader = DataLoader(inpaintingDataset, batch_size=2,
    #                     shuffle=True, num_workers=0)
    
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['mask'].size(),
    #         sample_batched['corrupted'].size())




    # THIS CODE IS FOR THE DATASET THAT LOADS THREE IMAGES SOURCES; ORIGINAL, CORRUPTED, AND MASK. NO TRANSFORMS REQUIRED

    inpaintingDataset = PreprocessedImageInpaintingDataset(r"C:\Users\eyad\Pictures\Images Datasets\2_cutouts_small_20px", extensions=["png"])
    dataloader = DataLoader(inpaintingDataset, batch_size=64, shuffle=True, num_workers=2)
    for i_batch, sample_batched in enumerate(dataloader):
        original, corrupted, mask =  sample_batched
        print(original.size())
        print(corrupted.size())
        print(mask.size())
        break
    img_grid=utils.make_grid(original, nrow=8, padding=4)
    # display result
    img = T.ToPILImage()(img_grid)
    img.show()