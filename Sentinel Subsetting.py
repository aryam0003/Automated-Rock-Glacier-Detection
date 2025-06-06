import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box
import os

def get_bounds(filepath):
    """Get bounds of a raster image in WGS84 (EPSG:4326)"""
    with rasterio.open(filepath) as src:
        bounds = src.bounds
        crs = src.crs
        wgs84_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
        return box(*wgs84_bounds), crs, bounds

def crop_to_overlap(s1_path, s2_path, output_folder_s1, output_folder_s2):
    """Crop Sentinel-1 and Sentinel-2 images to their overlapping region"""
    try:
        # Get bounding boxes and CRS
        s1_bounds, s1_crs, s1_orig_bounds = get_bounds(s1_path)
        s2_bounds, s2_crs, s2_orig_bounds = get_bounds(s2_path)
        
        # Print bounds for debugging
        print(f"S1 CRS: {s1_crs}")
        print(f"S2 CRS: {s2_crs}")
        print(f"S1 original bounds: {s1_orig_bounds}")
        print(f"S2 original bounds: {s2_orig_bounds}")
        print(f"S1 WGS84 bounds: {s1_bounds.bounds}")
        print(f"S2 WGS84 bounds: {s2_bounds.bounds}")
        
        # Compute intersection in WGS84
        overlap_area = s1_bounds.intersection(s2_bounds)
        
        if overlap_area.is_empty:
            print(f"No overlapping region found between the images.")
            return
            
        print(f"Overlapping Region (WGS84): {overlap_area.bounds}")
        
        def crop_and_save(input_path, output_folder, image_type):
            with rasterio.open(input_path) as src:
                try:
                    # Transform overlap geometry from WGS84 to the image's CRS
                    geom = transform_geom("EPSG:4326", src.crs, overlap_area.__geo_interface__)
                    
                    # Perform the crop
                    out_image, out_transform = mask(src, [geom], crop=True)
                    
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"cropped_{os.path.basename(input_path)}")
                    
                    with rasterio.open(output_path, "w", **out_meta) as dst:
                        dst.write(out_image)
                    print(f"Successfully saved cropped {image_type} image: {output_path}")
                    
                except ValueError as e:
                    print(f"Error cropping {image_type} image {input_path}: {str(e)}")
                    print(f"Image bounds in its CRS: {src.bounds}")
                    print(f"Overlap geometry in image CRS: {geom}")
                except Exception as e:
                    print(f"Unexpected error processing {image_type} image {input_path}: {str(e)}")
        
        # Crop both images
        crop_and_save(s1_path, output_folder_s1, "Sentinel-1")
        crop_and_save(s2_path, output_folder_s2, "Sentinel-2")
        
    except Exception as e:
        print(f"Error in crop_to_overlap: {str(e)}")

# Define paths
sentinel1_path = "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-1 SLC/S1A_IW_SLC__1SDV_20230921T124857_20230921T124924_050426_061280_7AC6_Orb_split_Stack_ifg_deb_TC_glcm.tif"
sentinel2_path = "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/S2A_MSIL2A_20241013T053801_N0511_R005_T43SFU_20241013T100347.SAFE/GRANULE/L2A_T43SFU_A048622_20241013T054552/IMG_DATA/R10m/T43SFU_20241013T053801_AOT_10m.jp2"

# Create output directories
output_folder_s1 = "cropped_sentinel1"
output_folder_s2 = "cropped_sentinel2"
os.makedirs(output_folder_s1, exist_ok=True)
os.makedirs(output_folder_s2, exist_ok=True)

# Process the files
if os.path.exists(sentinel1_path) and os.path.exists(sentinel2_path):
    print("Starting image processing...")
    crop_to_overlap(sentinel1_path, sentinel2_path, output_folder_s1, output_folder_s2)
else:
    if not os.path.exists(sentinel1_path):
        print(f"Sentinel-1 file not found: {sentinel1_path}")
    if not os.path.exists(sentinel2_path):
        print(f"Sentinel-2 file not found: {sentinel2_path}")
