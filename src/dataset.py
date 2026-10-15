import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from src.utils import convert_labelids_to_trainids

IMAGE_SUFFIX = '_leftImg8bit.png'
LABEL_SUFFIX = '_gtFine_labelIds.png'

image_transform = transforms.Compose([
    transforms.ToTensor() # tensor scaled to [0, 1]
])

label_transform = transforms.Compose([
    transforms.PILToTensor(), # tensor without scaling (stays 0-255)
    # remove dummy channel dimension and cast to int64 because PyTorch's CrossEntropyLoss requires this for segmentation. [1, H, W] -> [H, W]
    transforms.Lambda(lambda x: x.squeeze(0).long())
])

# random scale then crop to a fixed size
def random_crop_scale(image, mask, size, scale=(0.5, 2.0)):
    factor = random.uniform(*scale)

    # window in the original image that gets resized to the crop size
    height = round(size[0] / factor)
    width = round(size[1] / factor)

    top = random.randint(0, image.height - height)
    left = random.randint(0, image.width - width)

    image = TF.resized_crop(image, top, left, height, width, size, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resized_crop(mask, top, left, height, width, size, interpolation=TF.InterpolationMode.NEAREST)

    return image, mask

# random horizontal flip
def random_flip(image, mask):
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    return image, mask

# resize whole image to a fixed size
def resize(image, mask, size):
    image = TF.resize(image, size, interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST)

    return image, mask

# Docs: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CityscapesDataset(Dataset):
    def __init__(self, img_dir, image_transform=image_transform, label_transform=label_transform, crop_size=None, crop_scale=(0.5, 2.0), resize=None, flip=False):
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        # for training
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.resize = resize
        self.flip = flip

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

        # crop or resize
        if self.crop_size:
            image, mask = random_crop_scale(image, mask, self.crop_size, self.crop_scale)
        elif self.resize:
            image, mask = resize(image, mask, self.resize)

        if self.flip:
            image, mask = random_flip(image, mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)
        
        return image, mask