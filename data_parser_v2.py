import pandas as pd
import numpy as np

descriptor_table = pd.read_csv("Descriptors.csv")
descriptor_table[["sth", "B", "Patient ID", "Date"]] = descriptor_table["MRID"].str.split("_", expand=True)
descriptor_table.sort_values(by=["Patient ID", "Age"])
descriptor_table["Age-rounded"] = descriptor_table["Age"].round()

median_age = descriptor_table["Age-rounded"].median()

descriptor_table['Age_Group'] = descriptor_table['Age'] < median_age

# Map True to 'young' and False to 'old'
descriptor_table['Age_Group'] = descriptor_table['Age_Group'].map({True: 'young', False: 'old'})


from sklearn.model_selection import train_test_split
grouped = descriptor_table.groupby('Patient ID')
print(grouped)
# Extract unique Patient IDs
unique_patient_ids = descriptor_table['Patient ID'].unique()

# Shuffle the list of unique Patient IDs
np.random.shuffle(unique_patient_ids)

# Calculate the number of IDs for training and testing
total_ids = len(unique_patient_ids)
train_size = int(total_ids * 0.8)

# Assign 80% of IDs to train and 20% to test
train_ids = unique_patient_ids[:train_size]
test_ids = unique_patient_ids[train_size:]

# Create DataFrames for train and test
train_df = pd.DataFrame({'Patient ID': train_ids, 'dataset': 'train'})
test_df = pd.DataFrame({'Patient ID': test_ids, 'dataset': 'test'})

# Concatenate train and test DataFrames
full_df = pd.concat([train_df, test_df])

# Merge with the original DataFrame to assign 'dataset' to each row
result_df = pd.merge(descriptor_table, full_df, on='Patient ID', how='left')

# Display or save the resulting DataFrame
print(result_df)


# separate into train and test df
# Filter rows for train and test datasets
train_df = result_df[result_df['dataset'] == 'train']
test_df = result_df[result_df['dataset'] == 'test']
train_df

# discard longitudinal data in the train_df by only keeping the first scan
train_df = train_df.groupby('Patient ID').apply(lambda x: x.loc[x['Age'].idxmin()]).reset_index(drop=True)

# split train_df into young_train_df and old_train_df
young_train_df = train_df[train_df["Age_Group"] == "young"].reset_index(drop=True)
old_train_df = train_df[train_df["Age_Group"] == "old"].reset_index(drop=True)

# Filter rows for young and old age groups in the test set
young_test_df = test_df[test_df["Age_Group"] == "young"].reset_index(drop=True)
old_test_df = test_df[test_df["Age_Group"] == "old"].reset_index(drop=True)

import numpy as np
import os
import nibabel as nib
yng_imgs_tr = []
yng_age_tr = []
yng_AD_tr = []

yng_imgs_te = []
yng_age_te = []
yng_AD_te = []

old_imgs_tr = []
old_age_tr = []
old_AD_tr = []

old_imgs_te = []
old_age_te = []
old_AD_te = []


# yng_imgs_tr 
# get the list of 
yng_imgs_tr_list = young_train_df["MRID"]
# for i in range(10):
for i in range(len(yng_imgs_tr_list)):
    # get filename
    cur_img_file = nib.load(os.path.join('adni_data', yng_imgs_tr_list[i], yng_imgs_tr_list[i]+"_MNI152_registered.nii.gz"))
    cur_img_np = cur_img_file.get_fdata()
    cur_img_np_sliced = cur_img_np[:,int(cur_img_np.shape[1]/2-30):int(cur_img_np.shape[1]/2+30),:]
    slice_thickness = cur_img_np_sliced.shape[1]
    for j in range(slice_thickness):
        cur_slice = cur_img_np_sliced[:,j,:]
        # normalize the slice
        cur_slice = (cur_slice / np.max(cur_slice))  * 2 - 1
        yng_imgs_tr.append(cur_slice)
        cur_age = young_train_df[young_train_df["MRID"] == yng_imgs_tr_list[i]]["Age-rounded"].values
        yng_age_tr.append(cur_age[0])
        cur_condition = young_train_df[young_train_df["MRID"] == yng_imgs_tr_list[i]]["Diagnosis"].values
        yng_AD_tr.append(cur_condition[0])
        
yng_imgs_te_list = young_test_df["MRID"]
# for i in range(10):
for i in range(len(yng_imgs_te_list)):
    # get filename
    cur_img_file = nib.load(os.path.join('adni_data', yng_imgs_te_list[i], yng_imgs_te_list[i]+"_MNI152_registered.nii.gz"))
    cur_img_np = cur_img_file.get_fdata()
    cur_img_np_sliced = cur_img_np[:,int(cur_img_np.shape[1]/2-30):int(cur_img_np.shape[1]/2+30),:]
    slice_thickness = cur_img_np_sliced.shape[1]
    for j in range(slice_thickness):
        cur_slice = cur_img_np_sliced[:,j,:]
        # normalize the slice
        cur_slice = (cur_slice / np.max(cur_slice))  * 2 - 1
        yng_imgs_te.append(cur_slice)
        cur_age = young_test_df[young_test_df["MRID"] == yng_imgs_te_list[i]]["Age-rounded"].values
        yng_age_te.append(cur_age)
        cur_condition = young_test_df[young_test_df["MRID"] == yng_imgs_te_list[i]]["Diagnosis"].values
        yng_AD_te.append(cur_condition[0])
        
old_imgs_tr_list = old_train_df["MRID"]
# for i in range(10):
for i in range(len(old_imgs_tr_list)):
    # get filename
    cur_img_file = nib.load(os.path.join('adni_data', old_imgs_tr_list[i], old_imgs_tr_list[i]+"_MNI152_registered.nii.gz"))
    cur_img_np = cur_img_file.get_fdata()
    cur_img_np_sliced = cur_img_np[:,int(cur_img_np.shape[1]/2-30):int(cur_img_np.shape[1]/2+30),:]
    slice_thickness = cur_img_np_sliced.shape[1]
    for j in range(slice_thickness):
        cur_slice = cur_img_np_sliced[:,j,:]
        # normalize the slice
        cur_slice = (cur_slice / np.max(cur_slice)) * 2 - 1
        old_imgs_tr.append(cur_slice)
        cur_age = old_train_df[old_train_df["MRID"] == old_imgs_tr_list[i]]["Age-rounded"].values
        old_age_tr.append(cur_age[0])
        cur_condition = old_train_df[old_train_df["MRID"] == old_imgs_tr_list[i]]["Diagnosis"].values
        old_AD_tr.append(cur_condition[0])
        
old_imgs_te_list = old_test_df["MRID"]
# for i in range(10):
for i in range(len(old_imgs_te_list)):
    # get filename
    cur_img_file = nib.load(os.path.join('adni_data', old_imgs_te_list[i], old_imgs_te_list[i]+"_MNI152_registered.nii.gz"))
    cur_img_np = cur_img_file.get_fdata()
    cur_img_np_sliced = cur_img_np[:,int(cur_img_np.shape[1]/2-30):int(cur_img_np.shape[1]/2+30),:]
    slice_thickness = cur_img_np_sliced.shape[1]
    for j in range(slice_thickness):
        cur_slice = cur_img_np_sliced[:,j,:]
        # normalize the slice
        cur_slice = (cur_slice / np.max(cur_slice)) * 2 - 1
        old_imgs_te.append(cur_slice)
        cur_age = old_test_df[old_test_df["MRID"] == old_imgs_te_list[i]]["Age-rounded"].values
        old_age_te.append(cur_age)
        cur_condition = old_test_df[old_test_df["MRID"] == old_imgs_te_list[i]]["Diagnosis"].values
        old_AD_te.append(cur_condition[0])

yng_imgs_tr_array = np.stack(yng_imgs_tr, axis = 0)
yng_imgs_te_array = np.stack(yng_imgs_te, axis = 0)
old_imgs_tr_array = np.stack(old_imgs_tr, axis = 0)
old_imgs_te_array = np.stack(old_imgs_te, axis = 0)


# transcode AD condition from text to ord vector
# Mapping from category to numpy array representation
category_mapping = {"CN": [0, 0], "MCI": [0, 1], "AD": [1, 1]}

# Generate the corresponding numpy array
yng_AD_tr_array = np.array([category_mapping[category] for category in yng_AD_tr])
yng_AD_te_array = np.array([category_mapping[category] for category in yng_AD_te])
old_AD_tr_array = np.array([category_mapping[category] for category in old_AD_tr])
old_AD_te_array = np.array([category_mapping[category] for category in old_AD_te])


print("Length of yng_age_tr:", len(yng_age_tr))
print("Length of yng_AD_tr:", len(yng_AD_tr))

print("Length of yng_age_te:", len(yng_age_te))
print("Length of yng_AD_te:", len(yng_AD_te))

print("Length of old_age_tr:", len(old_age_tr))
print("Length of old_AD_tr:", len(old_AD_tr))

print("Length of old_age_te:", len(old_age_te))
print("Length of old_AD_te:", len(old_AD_te))


print(yng_imgs_tr_array.shape)
print(yng_imgs_te_array.shape)
print(old_imgs_tr_array.shape)
print(old_imgs_te_array.shape)

from scipy import interpolate
import torch
import torch.nn.functional as F
print("resampling")

def resample_stack_pytorch(original_stack, new_width, new_height, device='cuda'):
    print(".")
    # Move the original stack to the specified device (GPU or CPU)
    original_stack_tensor = torch.from_numpy(original_stack).to(device)
    
    # Add an extra dimension for the channel to satisfy the shape requirement (N, C, H, W)
    original_stack_tensor = original_stack_tensor.unsqueeze(1)
    
    # Perform the interpolation
    resampled_stack_tensor = F.interpolate(original_stack_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    
    # Remove the channel dimension and move the tensor back to the CPU
    resampled_stack = resampled_stack_tensor.squeeze(1).cpu().numpy()

    return resampled_stack

# Usage of the PyTorch resample function
device = 'cuda'  # or 'cuda:0', 'cuda:1', etc., depending on the specific GPU device you want to use
new_width = 208
new_height = 160
yng_imgs_tr_array = resample_stack_pytorch(yng_imgs_tr_array, new_width, new_height, device)
yng_imgs_te_array = resample_stack_pytorch(yng_imgs_te_array, new_width, new_height, device)
old_imgs_tr_array = resample_stack_pytorch(old_imgs_tr_array, new_width, new_height, device)
old_imgs_te_array = resample_stack_pytorch(old_imgs_te_array, new_width, new_height, device)

# The rest of the code remains unchanged
print("Resampled young train image stack shape:", yng_imgs_tr_array.shape)
print("Resampled young test image stack shape:", yng_imgs_te_array.shape)
print("Resampled old train image stack shape:", old_imgs_tr_array.shape)
print("Resampled old test image stack shape:", old_imgs_te_array.shape)



# Specify the file name
output_file = 'img_output.npz'

# Save the numpy arrays into one file
np.savez(output_file, yng_imgs_tr_array=yng_imgs_tr_array, yng_imgs_te_array=yng_imgs_te_array, old_imgs_tr_array=old_imgs_tr_array, old_imgs_te_array=old_imgs_te_array)

print("Numpy arrays saved to", output_file)

# Specify the output file names
ad_output_file = 'AD_data.npz'
age_output_file = 'age_data.npz'

# Save AD lists into a numpy compressed archive file
np.savez(ad_output_file, 
         yng_AD_tr=np.array(yng_AD_tr),
         yng_AD_te=np.array(yng_AD_te),
         old_AD_tr=np.array(old_AD_tr),
         old_AD_te=np.array(old_AD_te))

# Save age lists into a numpy compressed archive file
np.savez(age_output_file, 
         yng_age_tr=np.array(yng_age_tr),
         yng_age_te=np.array(yng_age_te),
         old_age_tr=np.array(old_age_tr),
         old_age_te=np.array(old_age_te))

print("AD data saved to", ad_output_file)
print("Age data saved to", age_output_file)


# Specify the file name
output_file_100 = 'img_output_100.npz'

# Save the numpy arrays into one file with only 100 in the first dimension
np.savez(output_file_100, 
         yng_imgs_tr_array=yng_imgs_tr_array[:100,:,:],
         yng_imgs_te_array=yng_imgs_te_array[:100,:,:],
         old_imgs_tr_array=old_imgs_tr_array[:100,:,:],
         old_imgs_te_array=old_imgs_te_array[:100,:,:])

print("Numpy arrays with first dimension 100 saved to", output_file_100)

# Specify the output file names
ad_output_file_100 = 'AD_data_100.npz'
age_output_file_100 = 'age_data_100.npz'

# Save AD lists into a numpy compressed archive file with only 100 in the first dimension
np.savez(ad_output_file_100, 
         yng_AD_tr=np.array(yng_AD_tr[:100]),
         yng_AD_te=np.array(yng_AD_te[:100]),
         old_AD_tr=np.array(old_AD_tr[:100]),
         old_AD_te=np.array(old_AD_te[:100]))

# Save age lists into a numpy compressed archive file with only 100 in the first dimension
np.savez(age_output_file_100, 
         yng_age_tr=np.array(yng_age_tr[:100]),
         yng_age_te=np.array(yng_age_te[:100]),
         old_age_tr=np.array(old_age_tr[:100]),
         old_age_te=np.array(old_age_te[:100]))

print("AD data with first dimension 100 saved to", ad_output_file_100)
print("Age data with first dimension 100 saved to", age_output_file_100)
