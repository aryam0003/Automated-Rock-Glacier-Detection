import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

def reproject_and_resample(input_path, output_path, target_crs, target_resolution):
    """Reprojects and resamples a raster to a target CRS and resolution."""
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        transform = transform * rasterio.Affine(
            target_resolution / src.res[0], 0, 0, 0, target_resolution / src.res[1], 0
        )
        
        profile = src.profile
        profile.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest
                )

def stack_rasters(input_files, output_file):
    """Stacks multiple raster layers into a single multi-band raster."""
    layers = []
    meta = None
    
    for file in input_files:
        with rasterio.open(file) as src:
            if meta is None:
                meta = src.meta
                meta.update(count=len(input_files))
            layers.append(src.read(1))
    
    with rasterio.open(output_file, 'w', **meta) as dst:
        for idx, layer in enumerate(layers, start=1):
            dst.write(layer, idx)

def normalize_raster(input_file, output_file):
    """Normalizes raster data to the range 0-1."""
    with rasterio.open(input_file) as src:
        data = src.read(1).astype(float)
        data_min, data_max = np.min(data), np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        
        profile = src.profile
        profile.update(dtype='float32')
        
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(normalized_data, 1)

def main():
    # File paths
    sentinel2_files = ["/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/Output/S2A_MSIL2A_20241013T053801_N0511_R005_T43SFU_20241013T100347_20250102_121037/NDVI.tif", "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/Output/S2A_MSIL2A_20241013T053801_N0511_R005_T43SFU_20241013T100347_20250102_121037/SAVI.tif", "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/Output/S2A_MSIL2A_20241013T053801_N0511_R005_T43SFU_20241013T100347_20250102_121037/SWIR.tif"]
    sentinel1_files = ["/Users/aryamohite/Documents/Documents/IIRS/Sentinel-1 SLC/S1A_IW_SLC__1SDV_20230921T124857_20230921T124924_050426_061280_7AC6_Orb_split_Stack_ifg_deb_TC_glcm.tif"]

    # Output folder
    output_folder = "/Users/aryamohite/Documents/Documents/IIRS/Fusion Run"
    os.makedirs(output_folder, exist_ok=True)

    # Parameters
    target_crs = "EPSG:4326"  # Change if needed
    target_resolution = 10  # In meters

    # Reproject and resample all files
    resampled_files = []
    for file in sentinel2_files + sentinel1_files:
        output_path = os.path.join(output_folder, f"resampled_{os.path.basename(file)}")
        reproject_and_resample(file, output_path, target_crs, target_resolution)
        resampled_files.append(output_path)

    # Stack all resampled rasters
    stacked_output = os.path.join(output_folder, "fused_stack.tif")
    stack_rasters(resampled_files, stacked_output)

    # Normalize the stacked raster
    normalized_output = os.path.join(output_folder, "fused_stack_normalized.tif")
    normalize_raster(stacked_output, normalized_output)

    print(f"Fused and normalized raster saved to: {normalized_output}")

if __name__ == "__main__":
    main()
