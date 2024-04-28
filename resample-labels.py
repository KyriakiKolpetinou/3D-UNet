import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os

def resample_image(input_path, output_path, new_spacing, interpolator):
    sitk_image = sitk.ReadImage(str(input_path))
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()

    # Calculate new size, assuming isotropic pixels
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    # Calculate physical size of the images
    original_physical_size = [sz*spc for sz, spc in zip(original_size, original_spacing)]
    new_physical_size = [sz*spc for sz, spc in zip(new_size, new_spacing)]

    # Calculate the centers of the old and new physical images
    original_center = np.array(original_physical_size) / 2.0
    new_center = np.array(new_physical_size) / 2.0

    # Compute the translation needed to align the centers
    translation = original_center - new_center

    # Set up the resampling with the computed translation
    transform = sitk.TranslationTransform(3, translation)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetTransform(transform)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())

    resampled_sitk_image = resampler.Execute(sitk_image)

    # Write the resampled image to disk
    sitk.WriteImage(resampled_sitk_image, str(output_path))

# Resample masks
masks_dataset_path = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/labelsTr')
resampled_masks_output_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/labelsTr_resampled')
resampled_masks_output_dir.mkdir(parents=True, exist_ok=True)

# Use the same median spacings computed from the images for resampling masks
for file_name in os.listdir(masks_dataset_path):
    if file_name.endswith('.nii.gz') and not file_name.startswith('._'):
        input_path = masks_dataset_path / file_name
        output_path = resampled_masks_output_dir / file_name
        resample_image(str(input_path), str(output_path), median_spacings, sitk.sitkNearestNeighbor)
