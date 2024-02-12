import nibabel as nib
import numpy as np

def count_pixels_per_class(seg_path, num_classes):
    # Load the segmentation mask
    seg_data = nib.load(seg_path).get_fdata()

    # Initialize an array to store the count of pixels for each class
    num_pixels_per_class = np.zeros(num_classes)

    # Iterate through each class
    for class_idx in range(num_classes):
        # Count the pixels for the current class
        num_pixels = np.sum(seg_data == class_idx)
        num_pixels_per_class[class_idx] = num_pixels

    return num_pixels_per_class

# Example usage
seg_path = "D:\\minor test1\\Data_Train_Test\\TrainSegmentation\\Case_00001.nii.gz"
num_classes = 5  # Assuming 5 classes including background
num_pixels_per_class = count_pixels_per_class(seg_path, num_classes)

print("Number of pixels per class:", num_pixels_per_class)
