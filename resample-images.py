import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os

def get_voxel_spacing(nifti_file_path):
    sitk_image = sitk.ReadImage(str(nifti_file_path))
    return sitk_image.GetSpacing()

def resample_image(input_path, output_path, new_spacing):
    sitk_image = sitk.ReadImage(str(input_path))
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    # Calculate the physical size of the old and new images
    original_physical_size = [sz*spc for sz, spc in zip(original_size, original_spacing)]
    new_physical_size = [sz*spc for sz, spc in zip(new_size, new_spacing)]

    # Calculate the center of the old and new physical images
    original_center = np.array(original_physical_size) / 2.0
    new_center = np.array(new_physical_size) / 2.0

    # Compute the translation vector
    translation = original_center - new_center

    # Create a transformation that will align the centers of the images
    transform = sitk.TranslationTransform(3, translation)

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampled_sitk_image = resampler.Execute(sitk_image)

    # Write the resampled image to the specified output path
    sitk.WriteImage(resampled_sitk_image, str(output_path))

# Path to your dataset directory
dataset_path = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr')

# Path to the output directory where resampled images will be saved
output_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr_resampled1')
output_dir.mkdir(parents=True, exist_ok=True)

# Calculate median voxel spacings
nifti_file_paths = [f for f in os.listdir(dataset_path) if f.endswith('.nii.gz') and not f.startswith('._')]
voxel_spacings = [get_voxel_spacing(str(dataset_path / f)) for f in nifti_file_paths]
spacings_array = np.array(voxel_spacings)
median_spacings = np.median(spacings_array, axis=0)

# Resample and save all valid images
for file_name in nifti_file_paths:
    input_path = dataset_path / file_name
    output_path = output_dir / file_name
    resample_image(str(input_path), str(output_path), median_spacings)
