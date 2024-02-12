import os
import glob
import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

def resample_nifti(input_path, output_path, target_num_slices, is_label=False):
    # Load the NIfTI file
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()

    # Get the current number of slices
    current_num_slices = data.shape[-1]

    # Create an interpolation function for resampling
    x_original = np.linspace(0, 1, current_num_slices)
    x_resampled = np.linspace(0, 1, target_num_slices)

    if is_label:
        # Use nearest-neighbor interpolation for label data
        interpolation_func = interp1d(x_original, data, kind='nearest', axis=-1, fill_value="extrapolate")
    elif is_label==False:
        # Use linear interpolation for image data
        interpolation_func = interp1d(x_original, data, kind='linear', axis=-1, fill_value="extrapolate")

    # Resample the data using the interpolation function
    resampled_data = interpolation_func(x_resampled)

    # Create a new NIfTI image with resampled data
    new_nifti_img = nib.Nifti1Image(resampled_data, nifti_img.affine)

    # Ensure the output path has the .nii.gz extension
    if not output_path.endswith('.nii.gz'):
        output_path += '.nii.gz'

    # Save the resampled NIfTI file
    nib.save(new_nifti_img, output_path)

# Specify input and output directories
input_folder = 'D:\\minor test1\\labels'
output_folder = 'D:\\minor test1\\Data_Train_Test\\TrainSegmentation'
target_num_slices = 50  # Adjust this value to the desired number of slices

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all .nii.gz files in the input folder
input_files = glob.glob(os.path.join(input_folder, '*.nii.gz'))

for input_file in input_files:
   
   

    # Create output file path based on the input file name
    output_file = os.path.join(output_folder, os.path.basename(input_file))

    # Resample the NIfTI file
    resample_nifti(input_file, output_file, target_num_slices, is_label=True)
