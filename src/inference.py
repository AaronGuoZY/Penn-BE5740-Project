import torch
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib
import os
from models import Generator
from data_loader import CustomImageDataset

# Setup
model_path = "/home/aarongzy/class/src/trained_models/generator.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_gen = Generator().to(device)
dataset = CustomImageDataset("/home/aarongzy/class/src/data")
batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
trained_gen.load_state_dict(torch.load(model_path, map_location=device))

# Create a directory to save generated images
output_dir = "/home/aarongzy/class/src/generated_nifti_imagessx"
os.makedirs(output_dir, exist_ok=True)

# Inference and save images
for batch_idx, (xi, yo, ad, ao, ho) in enumerate(train_loader):
    xi = xi.float().to(device)
    # ho = ho.float().to(device)
    ho = ho.float().to(device).squeeze(-1)
    ad = ad.float().to(device)
    current_batch_size = xi.size(0)  # Get the size of the current batch
    xi = xi.view(current_batch_size, 1, 208, 160)
    yo = yo.view(current_batch_size, 1, 208, 160)
    print(xi.shape)
    print(ho.shape)
    print(ad.shape)

    with torch.no_grad():
        generated_images = trained_gen(xi, ho, ad)

    # Move to CPU
    generated_images = generated_images.cpu()

    # Convert generated images to NIfTI and save
    for idx, image in enumerate(generated_images):
        image_np = image.numpy()  # Convert to numpy array
        image_nifti = nib.Nifti1Image(image_np, affine=np.eye(4))  # Create a NIfTI image; uses an identity matrix for affine
        filename = os.path.join(output_dir, f"batch{batch_idx}_img{idx}.nii")
        nib.save(image_nifti, filename)

print("All generated images have been saved in NIfTI format.")
