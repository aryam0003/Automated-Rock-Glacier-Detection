import os
import datetime
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Define input paths
coherence_path = "/Volumes/T7 Shield/IIRS/TanDEM-X 30m/._merged_EGM.tif"
glcm_path = "/Volumes/T7 Shield/IIRS/TanDEM-X 30m/._merged_W84.tif"  # Can be a directory or a single .tif file

# Output directories with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_coherence_dir = f"output_coherence_{timestamp}"
output_glcm_dir = f"output_glcm_{timestamp}"

os.makedirs(output_coherence_dir, exist_ok=True)
os.makedirs(output_glcm_dir, exist_ok=True)

# GLCM Band Names
glcm_bands = {
    1: "Contrast",
    2: "Dissimilarity",
    3: "Homogeneity",
    4: "ASM",
    5: "Energy",
    6: "Max",
    7: "Entropy",
    8: "GLCM_Mean",
    9: "GLCM_Variance",
    10: "GLCM_Correlation"
}

# Function to crop and save images
def crop_and_save(image_path, output_dir, is_glcm=False):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    print(f"🔄 Processing: {image_path}")

    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        tile_size = 1024
        img_name = os.path.splitext(os.path.basename(image_path))[0]

        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                w = min(tile_size, width - i)
                h = min(tile_size, height - j)
                window = Window(i, j, w, h)

                meta = src.meta.copy()
                meta.update({"width": w, "height": h, "transform": src.window_transform(window)})

                if is_glcm:
                    # Process each GLCM band separately
                    for band in range(1, min(src.count, 10) + 1):  # Max 10 bands for safety
                        cropped_image = src.read(band, window=window)
                        band_name = glcm_bands.get(band, f"Band_{band}")  # Default to "Band_X"

                        crop_filename = f"{img_name}_{band_name}_{i}_{j}_{timestamp}.tif"
                        crop_filepath = os.path.join(output_dir, crop_filename)

                        meta.update({"count": 1})  # Save only one band
                        with rasterio.open(crop_filepath, "w", **meta) as dest:
                            dest.write(cropped_image, 1)

                else:
                    # Process single-band coherence image
                    cropped_image = src.read(window=window)

                    crop_filename = f"{img_name}_{i}_{j}_{timestamp}.tif"
                    crop_filepath = os.path.join(output_dir, crop_filename)

                    with rasterio.open(crop_filepath, "w", **meta) as dest:
                        dest.write(cropped_image)

# Function to process directories or single files
def process_images(input_path, output_dir, is_glcm=False):
    if os.path.isdir(input_path):  # If directory
        print(f"📂 Processing directory: {input_path}")
        tif_files = [f for f in os.listdir(input_path) if f.endswith(".tif")]
        if not tif_files:
            print("⚠️ No .tif files found in directory!")
            return
        for file in tqdm(tif_files):
            crop_and_save(os.path.join(input_path, file), output_dir, is_glcm)
    elif os.path.isfile(input_path):  # If single file
        print(f"📄 Processing single file: {input_path}")
        crop_and_save(input_path, output_dir, is_glcm)
    else:
        print(f"❌ Invalid path: {input_path}")

# Process Coherence Images
print("\n🚀 Processing Coherence Images...")
process_images(coherence_path, output_coherence_dir, is_glcm=False)

# Process GLCM Images
print("\n🚀 Processing GLCM Images...")
process_images(glcm_path, output_glcm_dir, is_glcm=True)

print("\n✅ Cropping completed successfully!")
