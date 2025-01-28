import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class Sentinel2Processor:
    def __init__(self, safe_path: str):
        """
        Initialize the Sentinel-2 image processor.

        Args:
            safe_path (str): Path to the .SAFE directory.
        """
        self.safe_path = Path(safe_path)
        self.bands = {}
        self.indices = {}

    def find_band_path(self, band_name: str) -> str:
        """Find the correct path for a given band."""
        granule_dir = list(self.safe_path.glob('GRANULE/L2A_*'))[0]
        resolution_map = {
            'B02': 'R10m',  # Blue - 10m
            'B03': 'R10m',  # Green - 10m
            'B04': 'R10m',  # Red - 10m
            'B08': 'R10m',  # NIR - 10m
            'B11': 'R20m',  # SWIR1 - 20m
            'B12': 'R20m'   # SWIR2 - 20m
        }

        img_data_path = granule_dir / 'IMG_DATA'
        resolution = resolution_map[band_name]
        band_pattern = f'*_{band_name}_*.jp2'
        band_files = list(img_data_path.glob(f'{resolution}/{band_pattern}'))

        if not band_files:
            raise FileNotFoundError(f"Could not find {band_name} in {img_data_path}")

        return str(band_files[0])

    def load_bands(self):
        """Load required Sentinel-2 bands."""
        required_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

        for band_name in required_bands:
            band_path = self.find_band_path(band_name)
            with rasterio.open(band_path) as src:
                data = src.read(1).astype(float)
                
                # Resample 20m bands to 10m resolution
                if band_name in ['B11', 'B12']:
                    target_shape = (src.height * 2, src.width * 2)
                    data = src.read(
                        1,
                        out_shape=target_shape,
                        resampling=Resampling.bilinear
                    )
                self.bands[band_name] = data

                # Save metadata from B04 as a reference
                if band_name == 'B04':
                    self.meta = src.meta

        print("Bands loaded successfully")

    def calculate_ndvi(self):
        """Calculate NDVI."""
        red = self.bands['B04']
        nir = self.bands['B08']
        self.indices['NDVI'] = (nir - red) / (nir + red + 1e-6)

    def calculate_savi(self, L=0.5):
        """Calculate SAVI."""
        red = self.bands['B04']
        nir = self.bands['B08']
        self.indices['SAVI'] = ((nir - red) / (nir + red + L)) * (1 + L)

    def calculate_swir(self):
        """Calculate SWIR."""
        nir = self.bands['B08']
        swir1 = self.bands['B11']
        self.indices['SWIR'] = (nir - swir1) / (nir + swir1 + 1e-6)

    def save_index(self, index_name: str, output_path: str):
        """Save the specified index to a GeoTIFF file."""
        if index_name not in self.indices:
            raise ValueError(f"{index_name} index is not available.")
        
        data = self.indices[index_name].astype(np.float32)
        meta = self.meta.copy()
        meta.update({
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "compress": "lzw"
        })
        
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data, 1)
        
        print(f"{index_name} saved to {output_path}")

    def plot_index(self, index_name: str, cmap='viridis'):
        """Plot a calculated index."""
        if index_name not in self.indices:
            raise ValueError(f"Index {index_name} not calculated.")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.indices[index_name], cmap=cmap)
        plt.colorbar(label=index_name)
        plt.title(index_name)
        plt.axis('off')
        plt.show()

    def create_rgb_image(self, output_path: str):
        """Create and save an RGB composite image."""
        blue = self.bands['B02']
        green = self.bands['B03']
        red = self.bands['B04']

        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.clip((rgb / 3000) * 255, 0, 255).astype(np.uint8)

        meta = self.meta.copy()
        meta.update({"count": 3, "dtype": 'uint8'})

        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)

        print(f"RGB composite image saved to {output_path}")


def process_sentinel_image(safe_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    processor = Sentinel2Processor(safe_dir)

    try:
        processor.load_bands()
        processor.calculate_ndvi()
        processor.calculate_savi()
        processor.calculate_swir()

        processor.save_index('NDVI', os.path.join(output_dir, 'ndvi.tif'))
        processor.save_index('SAVI', os.path.join(output_dir, 'savi.tif'))
        processor.save_index('SWIR', os.path.join(output_dir, 'swir.tif'))

        processor.create_rgb_image(os.path.join(output_dir, 'rgb_composite.tif'))

        processor.plot_index('NDVI')
        processor.plot_index('SAVI')
        processor.plot_index('SWIR', cmap='coolwarm')

        print("Processing completed successfully.")

    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    safe_directory = "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/S2A_MSIL1C_20230926T052651_N0509_R105_T43SGT_20230926T071719.SAFE"
    output_directory = "/Users/aryamohite/Documents/Documents/IIRS/Sentinel-2A/Output"

    process_sentinel_image(safe_directory, output_directory)