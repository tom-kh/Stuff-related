from PIL import Image
import os

def convert_images_to_jpg(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over the files and convert them to JPG
    for file in files:
        if file.endswith('.png'):
            # Open the image file
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path)

            # Convert the image to JPG format
            jpg_path = os.path.splitext(img_path)[0] + '.jpg'
            img.convert('RGB').save(jpg_path, 'JPEG')
            #print(f"Converted {file} to JPG format.")

            # Close the image
            img.close()

# Specify the folder paths
folder1_path = '/media/auto/3adbbc9c-0e68-4197-ad21-692a439596e7/LIG_Dataset/Dongho/0mat/LIG_sync_2/LIG_5_3/SWIR'
folder2_path = '/media/auto/3adbbc9c-0e68-4197-ad21-692a439596e7/LIG_Dataset/Dongho/0mat/LIG_sync_2/LIG_5_3/SWIR'

# Call the function to convert images to JPG format for each folder
convert_images_to_jpg(folder1_path)
convert_images_to_jpg(folder2_path)
