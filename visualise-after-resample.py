import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
import numpy as np
def visualize_overlay(image_path, mask_path, slice_idx=None):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    if slice_idx is None:
        slice_idx = np.random.randint(0,image_array.shape[0]-1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_array[slice_idx], cmap='gray')
    ax[0].set_title('Image Slice')
    ax[0].axis('off')

    ax[1].imshow(image_array[slice_idx], cmap='gray')
    ax[1].imshow(mask_array[slice_idx], alpha=0.5, cmap='jet')  # Overlay with transparency
    ax[1].set_title('Mask Overlay')
    ax[1].axis('off')

    plt.show()

# Paths to the resampled images and masks
resampled_images_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr_resampled1')
resampled_masks_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/labelsTr_resampled')

# List of specific images to visualize
specific_images = ['liver_59.nii.gz','liver_93.nii.gz','liver_1.nii.gz', 'liver_2.nii.gz', 'liver_3.nii.gz', 'liver_4.nii.gz', 'liver_5.nii.gz','liver_66.nii.gz', 'liver_99.nii.gz','liver_58.nii.gz','liver_43.nii.gz','liver_101.nii.gz','liver_130.nii.gz','liver_77.nii.gz']

for image_name in specific_images:
    # Assuming mask filename can be derived from image filename
    mask_name = image_name.replace('image', 'mask')  # Adjust based on your naming convention

    image_path = resampled_images_dir / image_name
    mask_path = resampled_masks_dir / mask_name

    if image_path.exists() and mask_path.exists():
        visualize_overlay(image_path, mask_path)
    else:
        print(f"Image or mask not found for {image_name}")
