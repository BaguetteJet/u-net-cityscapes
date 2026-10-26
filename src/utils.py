import torch
import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
from cityscapesscripts.helpers.labels import labels

# Official Cityscapes benchmark uses 19 classes for evaluation, remaining classes are ignored
# labelId to trainId mapping (19 classes + ignore=255)
# Benchmark: https://www.cityscapes-dataset.com/benchmarks/
# Source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

# create lookup table for fast conversion
lookup_table = np.full(256, 255, dtype=np.uint8) # default to 255, uint8 = 0-255
for label in labels:
    if 0 <= label.id < len(lookup_table):
        lookup_table[label.id] = label.trainId

def convert_labelids_to_trainids(label_image: Image.Image) -> Image.Image:
    mask = np.asarray(label_image, dtype=np.uint8) # convert mask image to numpy array for lookup table
    mask = lookup_table[mask] # use lookup table to convert labels
    mask = Image.fromarray(mask) # convert back to image
    return mask

def printlabels():
    for label in labels:
        if label.trainId != 255:
            print(f"{label.id:2} -> {label.trainId:2}: {label.name}")

# fails if in notebook so put here
def mask_to_long_tensor(x):
    return x.squeeze(0).long()

# save png of prediciton
def save_prediction(model, image, epoch, output_dir, device):
    model.eval()

    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device))
        pred = pred.argmax(1).squeeze().cpu().numpy()

    # colour predictions
    colours = np.zeros((256, 3), dtype=np.uint8)
    for label in labels:
        if label.trainId != 255:
            colours[label.trainId] = label.color

    pred = colours[pred]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(pred.astype(np.uint8)).save(f"{output_dir}/epoch_{epoch:03d}.png")

def make_gif(output_dir, gif_path):
    files = sorted(Path(output_dir).glob("epoch_*.png"))
    imageio.mimsave(gif_path, [imageio.imread(f) for f in files], duration=0.5, loop=0)
