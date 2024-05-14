import pandas as pd
# import data files
young_train_df = pd.read_csv("young_train_df.csv")
young_test_df = pd.read_csv("young_test_df.csv")
old_train_df = pd.read_csv("old_train_df.csv")
old_test_df = pd.read_csv("old_test_df.csv")
# Assuming young_train_df and old_train_df are already loaded DataFrames

# Function to randomly select a matching row from old_train_df based on constraints
def select_matching_row(row):
    if row['Diagnosis'] == 'CN':
        # Filter old_train_df based on constraints
        filtered_old_df = old_train_df[(old_train_df['Diagnosis'] == 'CN') | (old_train_df['Diagnosis'] == 'MCI') | (old_train_df['Diagnosis'] == 'AD')]
    elif row['Diagnosis'] == 'MCI':
        # Filter old_train_df based on constraints
        filtered_old_df = old_train_df[(old_train_df['Diagnosis'] == 'MCI') | (old_train_df['Diagnosis'] == 'AD')]
    else:
        # Filter old_train_df based on constraints
        filtered_old_df = old_train_df[old_train_df['Diagnosis'] == 'AD']
    
    # Randomly select a row from filtered_old_df
    selected_row = filtered_old_df.sample(n=1)
    
    # Return the MRID, Age, and Diagnosis of the selected row
    return selected_row['MRID'].values[0], selected_row['Age-rounded'].values[0], selected_row['Diagnosis'].values[0]

# Apply the function to each row in young_train_df and expand the returned results into multiple columns
result = young_train_df.apply(select_matching_row, axis=1, result_type='expand')
young_train_df['MRID_pair'], young_train_df['OldAgeRounded'], young_train_df['OldDiagnosis'] = result[0], result[1], result[2]


import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import nibabel as nib
import numpy as np
import torch

# Assuming 'young_train_df' is already loaded and contains columns 'MRID' and 'MRID_pair'
# Path to the directory where folders named by MRID are stored
base_image_path = 'adni_data'

# Function to load a NIfTI image from a specific folder
def load_nifti_image(mrid):
    folder_path = os.path.join(base_image_path, mrid, mrid+"_MNI152_registered.nii.gz")
    try:
        # We assume the file follows a naming convention such as 'image.nii.gz'
        image_path = os.path.join(folder_path)  # Adjust the file name as needed
        image = nib.load(image_path)
        image = image.get_fdata()
        return image
    except FileNotFoundError:
        print(f"No image found for MRID {mrid} in {folder_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading NIfTI image for MRID {mrid}: {str(e)}")
        return None

# Function to encode the diagnosis based on predefined category mapping
def encode_diagnosis(diagnosis):
    category_mapping = {"CN": [0, 0], "MCI": [0, 1], "AD": [1, 1]}
    # Fetch the list from the mapping or default to [None, None] if diagnosis is not found
    encoded_list = category_mapping.get(diagnosis, [None, None])
    # Convert list to a 2x1 NumPy array
    return np.array(encoded_list).reshape(2, 1)

def age_vector(age):
    vector = np.zeros(100, dtype=int)
    if age < 100:  # If age is 100 or more, the vector will be all 1s
        vector[100-age:] = 1
    else:
        vector[:] = 1
    return vector


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Iterate through each row in the DataFrame

# Ensure the directory exists or create it if it does not
if not os.path.exists("src/data"):
    os.makedirs("src/data")

for index, row in young_train_df.iterrows():
    young_image = load_nifti_image(row['MRID'])
    old_image = load_nifti_image(row['MRID_pair'])
    young_age = int(row["Age-rounded"])
    print(type(young_image))
    old_age = int(row["OldAgeRounded"])
    young_condition = encode_diagnosis(row["Diagnosis"])
    old_condition = encode_diagnosis(row["OldDiagnosis"])

    age_difference = abs(old_age - young_age)

    old_age_vector = age_vector(old_age)
    age_difference_vector = age_vector(age_difference)

    # Normalize and resize images if the images are loaded and in numpy array format
    if isinstance(young_image, np.ndarray) and isinstance(old_image, np.ndarray):
        # Compute the 99.5th percentile intensity value
        old_image_percentile = np.percentile(old_image, 99.5)
        young_image_percentile = np.percentile(young_image, 99.5)

        # Rescale the intensities
        old_image = np.clip(old_image, 0, old_image_percentile)
        young_image = np.clip(young_image, 0, young_image_percentile)
        # print(old_image.shape)

        # Normalize pixel values to range [-1, 1]
        old_image_normalized = (old_image / np.abs(np.max(old_image))) * 2 - 1
        young_image_normalized = (young_image / np.abs(np.max(young_image))) * 2 - 1

        # Convert numpy arrays to torch tensors and send them to CUDA device
        old_image_tensor = torch.tensor(old_image_normalized, device=device, dtype=torch.float32)
        young_image_tensor = torch.tensor(young_image_normalized, device=device, dtype=torch.float32)

        # Slice and process image tensors on GPU
        slice_count = old_image_tensor.shape[2]
        central_start = max(slice_count // 2 - 2, 0)
        central_end = min(central_start + 2, slice_count)
        young_image_central_slices = young_image_tensor[:, :, central_start:central_end]
        old_image_central_slices = old_image_tensor[:, :, central_start:central_end]

        resize = torch.nn.functional.interpolate
        for i in range(young_image_central_slices.shape[2]):
            print(".")
            young_slice = young_image_central_slices[:, :, i].unsqueeze(0).unsqueeze(0)
            old_slice = old_image_central_slices[:, :, i].unsqueeze(0).unsqueeze(0)

            # Resize images on GPU
            young_image_reshaped = resize(young_slice, size=(208, 160), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            old_image_reshaped = resize(old_slice, size=(208, 160), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            # Transfer tensors back to CPU for saving
            young_image_reshaped_cpu = young_image_reshaped.cpu().numpy()
            old_image_reshaped_cpu = old_image_reshaped.cpu().numpy()

            # Save the processed data
            output_file = f"{row['MRID']}_{row['MRID_pair']}_{str(i)}.npz"
            output_path = os.path.join("src", "data", output_file)
            np.savez(output_path, xi=young_image_reshaped_cpu, yo=old_image_reshaped_cpu, ad=age_difference_vector, ao=old_age_vector, ho=old_condition)
            print("Data saved to", output_path)



