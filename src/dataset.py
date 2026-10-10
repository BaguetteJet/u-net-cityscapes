import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from cityscapesscripts.helpers.labels import labels

IMAGE_SUFFIX = '_leftImg8bit.png'
LABEL_SUFFIX = '_gtFine_labelIds.png'

# Official Cityscapes benchmark uses 19 classes for evaluation, remaining classes are ignored
# labelId to trainId mapping (19 classes + ignore=255)
# Benchmark: https://www.cityscapes-dataset.com/benchmarks/
# Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

# create lookup table for fast conversion
lookup_table = np.full(256, 255, dtype=np.uint8) # default to 255, uint8 = 0-255
for label in labels:
    if 0 <= label.id < len(lookup_table):
        lookup_table[label.id] = label.trainId

def convert_labelids_to_trainids(label_image):
    mask = np.asarray(label_image, dtype=np.uint8) # convert to numpy array for lookup table
    mask = lookup_table[mask] # use lookup table to convert labels
    mask = Image.fromarray(mask) # convert back to PIL Image
    return mask

def printlabels():
    for label in labels:
        if label.trainId != 255:
            print(f"{label.id:2} -> {label.trainId:2}: {label.name}")

# Docs: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.images = sorted(file for file in os.listdir(img_dir) if file.endswith(IMAGE_SUFFIX)) # image files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = img_path.replace(IMAGE_SUFFIX, LABEL_SUFFIX) # get matching mask

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = convert_labelids_to_trainids(mask)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

def mask_to_long_tensor(x):
    return x.squeeze(0).long()