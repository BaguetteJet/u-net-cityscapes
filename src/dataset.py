import os
from PIL import Image
from torch.utils.data import Dataset
from src.utils import convert_labelids_to_trainids

IMAGE_SUFFIX = '_leftImg8bit.png'
LABEL_SUFFIX = '_gtFine_labelIds.png'

# Docs: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, image_transform=None, label_transform=None):
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.label_transform = label_transform

        self.images = sorted(file for file in os.listdir(img_dir) if file.endswith(IMAGE_SUFFIX)) # image files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = img_path.replace(IMAGE_SUFFIX, LABEL_SUFFIX) # get matching mask

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # labelId to trainId mapping (19 classes + ignore=255)
        mask = convert_labelids_to_trainids(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)
        
        return image, mask