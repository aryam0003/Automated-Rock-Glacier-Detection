import os
import glob
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from datetime import datetime
import re
from pathlib import Path

def clip_with_shapefile(tif_path, shapefile_path, output_dir):
    """
    Clip a TIF file with shapes from a shapefile and save the result
    
    Parameters:
    -----------
    tif_path : str
        Path to the input TIF file
    shapefile_path : str
        Path to the shapefile containing glacier polygons
    output_dir : str
        Directory to save clipped outputs
    """
    # Get the original filename without extension
    original_filename = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Extract timestamp from filename (assuming format contains a date pattern)
    # Example: sentinel_20210615.tif or s2_20210615_1024x1024.tif
    timestamp_match = re.search(r'_(\d{8})_?', os.path.basename(tif_path))
    if timestamp_match:
        timestamp = timestamp_match.group(1)
    else:
        # Use file modification date if no timestamp in filename
        timestamp = datetime.fromtimestamp(os.path.getmtime(tif_path)).strftime('%Y%m%d')
    
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Open the raster
    with rasterio.open(tif_path) as src:
        # Loop through each glacier in the shapefile
        for idx, glacier in gdf.iterrows():
            # Get glacier name/id
            glacier_name = glacier['name'] if 'name' in glacier else f"glacier_{idx}"
            
            # Get the geometry
            geom = [glacier.geometry]
            
            try:
                # Clip the raster with the glacier geometry
                clipped_img, clipped_transform = mask(src, geom, crop=True)
                
                # Copy the metadata
                meta = src.meta.copy()
                
                # Update metadata for the clipped raster
                meta.update({
                    "height": clipped_img.shape[1],
                    "width": clipped_img.shape[2],
                    "transform": clipped_transform
                })
                
                # Create output filename that keeps the original filename and adds glacier name
                output_filename = f"{original_filename}_{glacier_name}_{timestamp}.tif"
                output_path = os.path.join(output_dir, output_filename)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Write the clipped raster
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(clipped_img)
                
                print(f"Clipped {os.path.basename(tif_path)} with {glacier_name} and saved to {output_path}")
                
            except (ValueError, Exception) as e:
                print(f"Error processing {glacier_name} in {os.path.basename(tif_path)}: {e}")
                continue

def main():
    sentinel_data_dir = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices_cropped'  # Directory containing Sentinel-1 & 2 .tif files
    shapefile_path = "/Volumes/T7 Shield/IIRS/RG/RG_leh.shp"  # Path to rock glacier shapefile
    output_dir = '/Volumes/T7 Shield/IIRS/Sentinel-2_Indices_cropped_clipped'  # Output directory for clipped rasters
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all tif files (both S1 and S2)
    tif_files = glob.glob(os.path.join(sentinel_data_dir, '**', '*.tif'), recursive=True)
    
    print(f"Found {len(tif_files)} TIF files for processing")
    
    # Process each file
    for tif_file in tif_files:
        print(f"Processing: {os.path.basename(tif_file)}")
        clip_with_shapefile(tif_file, shapefile_path, output_dir)
    
    print("Clipping complete!")

if __name__ == "__main__":
    main()