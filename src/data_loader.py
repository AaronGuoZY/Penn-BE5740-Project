import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class CustomImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".npz")]
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_path  = self.image_files[idx]
        # Load the data from npz file
        data = np.load(image_path)
        # Extract arrays using the expected keys
        xi = data['xi']
        yo = data['yo']
        ad = data['ad']
        ao = data['ao']
        ho = data['ho']
        # Close the file to free up system resources
        data.close()
        return xi, yo, ad, ao, ho