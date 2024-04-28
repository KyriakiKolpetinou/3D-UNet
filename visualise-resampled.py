import matplotlib.pyplot as plt
import SimpleITK as sitk

def visualize_slice(image_path):
    image = sitk.ReadImage(str(image_path))
    array = sitk.GetArrayFromImage(image)
    slice_idx = array.shape[0] // 2  # Middle slice index
    print(array.shape)
    plt.imshow(array[slice_idx], cmap='gray')
    plt.axis('off')
    plt.show()

output_dir = Path('C:/Users/Kuria/Downloads/Task03_Liver/Task03_Liver/imagesTr_resampled1')  # Replace with your output directory path
resampled_files = list(output_dir.glob('*.nii.gz'))  # List the resampled files

# Visualize the first few images
for file_path in resampled_files[:12]:  # Change the number as needed
    visualize_slice(file_path)