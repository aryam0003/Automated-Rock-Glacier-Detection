import os
import datetime
import numpy as np
import tifffile
import shutil

def safe_remove_dir(folder_path):
    """
    Safely remove a directory, handling macOS hidden files
    """
    try:
        # First, remove any hidden macOS files
        for filename in os.listdir(folder_path):
            if filename.startswith('._'):
                try:
                    os.remove(os.path.join(folder_path, filename))
                except Exception as e:
                    print(f"Could not remove hidden file {filename}: {e}")
        
        # Then remove the directory
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        # If folder doesn't exist, that's fine
        pass
    except Exception as e:
        print(f"Error removing directory {folder_path}: {e}")

def crop_tiff_images(input_folder, output_folder, crop_size=1024):
    """
    Crop TIFF images in the input folder into 1024x1024 pixel boxes,
    with enhanced logging and multi-dimensional image handling.
    
    Args:
    input_folder (str): Path to the folder containing source TIFF images
    output_folder (str): Path to the base folder where cropped images will be saved
    crop_size (int): Size of the square crop (default 1024)
    """
    # Create base output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Timestamp for this batch of crops
    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each file in the input folder
    processed_files = 0
    total_crops = 0
    
    for filename in os.listdir(input_folder):
        # Skip macOS system files and non-TIFF files
        if filename.startswith('._') or not (filename.lower().endswith('.tif') or filename.lower().endswith('.tiff')):
            continue
        
        # Full path to the input image
        input_path = os.path.join(input_folder, filename)
        
        # Create a clean subfolder name (remove any problematic characters)
        base_filename = os.path.splitext(filename)[0]
        safe_filename = "".join(c for c in base_filename if c.isalnum() or c in ('-', '_'))
        file_output_folder = os.path.join(output_folder, safe_filename)
        
        # Safely remove existing folder if it exists
        safe_remove_dir(file_output_folder)
        
        # Create the output folder
        os.makedirs(file_output_folder)
        
        try:
            # Read the image 
            img = tifffile.imread(input_path)
            
            # Print image details for debugging
            print(f"\nProcessing {filename}")
            print(f"Image shape: {img.shape}")
            print(f"Image dtype: {img.dtype}")
            
            # Determine how to handle different image dimensions
            if img.ndim == 2:
                # Single 2D image
                crops = crop_image_slice(img, filename, 0, file_output_folder, batch_timestamp, crop_size)
                total_crops += crops
            elif img.ndim == 3:
                # Multiple scenarios for 3D images
                if img.shape[2] in [3, 4]:  # RGB or RGBA
                    # Treat as a single multi-channel image
                    crops = crop_multi_channel_image(img, filename, file_output_folder, batch_timestamp, crop_size)
                    total_crops += crops
                else:
                    # Multi-page or multi-slice image
                    for i in range(img.shape[0]):
                        crops = crop_image_slice(img[i], filename, i, file_output_folder, batch_timestamp, crop_size)
                        total_crops += crops
            elif img.ndim == 4:
                # Multi-page multi-channel image
                for i in range(img.shape[0]):
                    crops = crop_multi_channel_image(img[i], filename, file_output_folder, batch_timestamp, crop_size)
                    total_crops += crops
            else:
                print(f"Unsupported image dimension: {img.ndim}")
                continue
            
            processed_files += 1
            print(f"Completed processing {filename}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nSummary:")
    print(f"Total files processed: {processed_files}")
    print(f"Total crops generated: {total_crops}")

def crop_multi_channel_image(img, original_filename, output_folder, batch_timestamp, crop_size):
    """
    Crop a multi-channel image into 1024x1024 boxes.
    """
    # Get image dimensions
    height, width, channels = img.shape
    
    # Calculate number of crops in each dimension
    crops_height = height // crop_size
    crops_width = width // crop_size
    
    crops_generated = 0
    
    # Generate crops
    for y in range(crops_height):
        for x in range(crops_width):
            # Calculate crop coordinates
            y_start = y * crop_size
            x_start = x * crop_size
            
            # Extract crop
            crop = img[y_start:y_start+crop_size, x_start:x_start+crop_size, :]
            
            # Create output filename
            base_name = os.path.splitext(original_filename)[0]
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
            crop_filename = f"{safe_base_name}_crop_{y}_{x}_{batch_timestamp}.tif"
            output_path = os.path.join(output_folder, crop_filename)
            
            # Save crop
            tifffile.imwrite(output_path, crop)
            crops_generated += 1
    
    return crops_generated

def crop_image_slice(img_slice, original_filename, slice_index, output_folder, batch_timestamp, crop_size):
    """
    Crop a single 2D image slice into 1024x1024 boxes.
    """
    # Get image dimensions
    height, width = img_slice.shape
    
    # Calculate number of crops in each dimension
    crops_height = height // crop_size
    crops_width = width // crop_size
    
    crops_generated = 0
    
    # Generate crops
    for y in range(crops_height):
        for x in range(crops_width):
            # Calculate crop coordinates
            y_start = y * crop_size
            x_start = x * crop_size
            
            # Extract crop
            crop = img_slice[y_start:y_start+crop_size, x_start:x_start+crop_size]
            
            # Create output filename
            base_name = os.path.splitext(original_filename)[0]
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
            crop_filename = f"{safe_base_name}_slice{slice_index}_crop_{y}_{x}_{batch_timestamp}.tif"
            output_path = os.path.join(output_folder, crop_filename)
            
            # Save crop
            tifffile.imwrite(output_path, crop)
            crops_generated += 1
    
    return crops_generated

# Example usage
if __name__ == "__main__":
    input_folder = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices'  # Input folder path
    output_folder = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices_cropped'  # Output folder path
    
    crop_tiff_images(input_folder, output_folder)

print("TIFF Cropping Script is ready to use!")