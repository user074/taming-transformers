from PIL import Image
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # for progress bar
import traceback



def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

# Define the model_quantizer function here
def model_quantizer(batch_images):
    batch_images = batch_images.to(DEVICE)
    with torch.no_grad():  # Do not compute gradient since we are only doing inference
        z, a, idx = model16384.encode(batch_images * 2 - 1)
    indices = idx[-1]  # Assuming idx[-1] is the tensor containing the indices
    return indices.cpu()



class ImageFolderWithPaths(datasets.ImageFolder):
    # Override the __getitem__ method. This is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def process_and_save_indices(dataloader, output_dir, indices_per_image=256):
    try:
        # Iterate with an index to keep track of overall progress
        for batch_idx, (batch_images, _, paths) in enumerate(tqdm(dataloader, desc="Processing Images")):
            flat_indices = model_quantizer(batch_images)
            
            # Ensure we reshape the indices correctly based on the batch size
            batch_size = batch_images.size(0)
            if flat_indices.numel() != batch_size * indices_per_image:
                raise ValueError(f"Number of indices does not match expected size for batch {batch_idx}. "
                                 f"Got {flat_indices.numel()} indices for a batch size of {batch_size} images.")
            
            batch_indices = flat_indices.view(batch_size, indices_per_image)
            
            for img_idx, (path, index) in enumerate(zip(paths, batch_indices)):
                rel_path = os.path.relpath(path, start=dataloader.dataset.root)
                out_path = os.path.join(output_dir, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                out_file = os.path.splitext(out_path)[0] + '.npy'
                np.save(out_file, index.numpy())

                # Diagnostic print statement
                if not os.path.isfile(out_file):
                    print(f"Expected to save file but it's missing: {out_file}")
                else:
                    # print(f"File saved successfully: {out_file}")
                    pass

    except Exception as e:
        print(f"An error occurred in batch {batch_idx}, image {img_idx}: {e}")
        traceback.print_exc()  # This will print the stack trace of the error

if __name__ == "__main__":
    #read args
    args = sys.argv
    if len(args) != 3:
        print("Usage: python encode_images.py <path_to_imageNet> <path_to_output>")
        exit(1)
    imagenet_path = args[1]
    output_path = args[2]
    print(f"imagenet_path: {imagenet_path}")
    print(f"output_path: {output_path}")

    
    sys.path.append(".")
    # also disable grad to save memory
    torch.set_grad_enabled(False)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
    model16384 = load_vqgan(config16384, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)
    model16384.to(DEVICE).eval()

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Define your dataset using the custom class
    dataset = ImageFolderWithPaths(root= imagenet_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Specify the output directory where the indices will be saved
    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)

    # Process the dataset
    process_and_save_indices(dataloader, output_dir)
