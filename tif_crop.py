import os
import tifffile
import numpy as np
from PIL import Image

def convert_to_tiff(input_path, output_path):
    try:
        img = Image.open(input_path)
        img = img.convert("RGB")  # Ensure it's RGB
        img.save(output_path, format="TIFF")
        print(f"Converted {input_path} to valid TIFF format.")
        return output_path
    except Exception as e:
        print(f"Failed to convert {input_path}: {e}")
        return None

def process_tiff_files(input_folder, output_folder, crop_size=1024):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_crops = 0
    total_files = 0

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.tif', '.tiff')):
            print(f"Skipping non-TIFF file: {filename}")
            continue

        file_path = os.path.join(input_folder, filename)
        try:
            with tifffile.TiffFile(file_path) as tif:
                if not tif.is_imagej:
                    raise ValueError("Not a valid TIFF file")
                img = tif.asarray()
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
            # Attempt conversion for non-TIFF RGB images
            converted_path = convert_to_tiff(file_path, file_path + "_fixed.tif")
            if converted_path:
                file_path = converted_path  # Use the new TIFF file
                img = tifffile.imread(file_path)
            else:
                continue  # Skip if conversion fails

        print(f"Processing {filename}")
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        
        # Ensure the image is at least 2D
        if len(img.shape) < 2:
            print(f"Skipping {filename}: Invalid shape {img.shape}")
            continue
        
        # Crop and save
        height, width = img.shape[:2]
        for i in range(0, height, crop_size):
            for j in range(0, width, crop_size):
                crop = img[i:i+crop_size, j:j+crop_size]
                crop_filename = f"{filename}_crop_{i}_{j}.tif"
                tifffile.imwrite(os.path.join(output_folder, crop_filename), crop)
                total_crops += 1

        total_files += 1
        print(f"Completed processing {filename}")

    print("\nSummary:")
    print(f"Total files processed: {total_files}")
    print(f"Total crops generated: {total_crops}")
    print("TIFF Cropping Script is ready to use!")

# Example Usage
input_folder = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices'
output_folder = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices_cropped'
process_tiff_files(input_folder, output_folder)
