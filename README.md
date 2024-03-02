
# Multi-organ Abdomen Segmentation Using UNet and SegResNet

This is a comparison of Segmentation which is done to segment multiple organ from CT-images using UNet architecture and SegResNet.


## Dataset
The required Dataset is obtained from publicly availabe Dataset
provided by AbdomenCT-1K throug this form https://docs.google.com/forms/d/e/1FAIpQLSeuZ3yanPc0E-SxvYD2ZX8eu-BKxxdQT5GQUpyzfUeK39ytow/viewform


## Preprocessing
The availabe data is in nifti formate and the number of slices in the 3D volumes are uneven so we perfromed Interpolation

**For 3D Volume we use Linear Interpolation
**For the corresponding 3D labels we use nearest Interpolation to prevent mixing of labels
## Training  and testing (UNet)
This graph and output illustrates the progress of our Unet model over the course of training, showcasing key metrics. By monitoring both the training and testing phases, we gain insights into how well our model generalizes to new, unseen data.

![Alt text](https://github.com/dipesh0078/Multiorgan-segmentation-of-Abdomen-using-UNET-SegResNet/blob/main/training%20and%20testing/unet%20graph.png)

![Alt text](https://github.com/dipesh0078/Multiorgan-segmentation-of-Abdomen-using-UNET-SegResNet/blob/main/training%20and%20testing/unet%20jupyter%20output.png)
## Training and testing (SegresNet)
This is for SegresNet model with its output

![alt](https://github.com/dipesh0078/Multiorgan-segmentation-of-Abdomen-using-UNET-SegResNet/blob/main/training%20and%20testing/SegRestNet%20graph.png)

![alt](https://github.com/dipesh0078/Multiorgan-segmentation-of-Abdomen-using-UNET-SegResNet/blob/main/training%20and%20testing/SegResNet%20jupyter%20output.png)
