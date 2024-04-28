import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

def visualize_images_side_by_side(original_dir, resampled_dir, num_images=10):
    original_files = sorted([f for f in original_dir.glob('*.nii.gz') if not f.name.startswith('._')])
    resampled_files = sorted([f for f in resampled_dir.glob('*.nii.gz') if not f.name.startswith('._')])

    for orig_file, resamp_file in zip(original_files[:num_images], resampled_files[:num_images]):
        orig_img = sitk.ReadImage(str(orig_file))
        resamp_img = sitk.ReadImage(str(resamp_file))

        orig_array = sitk.GetArrayFromImage(orig_img)
        resamp_array = sitk.GetArrayFromImage(resamp_img)

        # Check multiple slices
        for slice_idx in range(0, orig_array.shape[0], orig_array.shape[0] // 10):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].imshow(orig_array[slice_idx], cmap='gray')
            axs[0].set_title(f'Original: {orig_file.name} Slice {slice_idx}')

            if slice_idx < resamp_array.shape[0]:
                axs[1].imshow(resamp_array[slice_idx], cmap='gray')
                axs[1].set_title(f'Resampled: {resamp_file.name} Slice {slice_idx}')
            else:
                axs[1].text(0.5, 0.5, 'No corresponding slice', horizontalalignment='center', verticalalignment='center')
                axs[1].set_title(f'Resampled: {resamp_file.name} Slice {slice_idx}')

            for ax in axs:
                ax.axis('off')

            plt.show()

# Replace these paths with the actual paths to your directories
original_images_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr')
resampled_images_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr_resampled1')

visualize_images_side_by_side(original_images_dir, resampled_images_dir)