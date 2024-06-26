from PIL import Image
import numpy as np
import os
import tifffile

# Convert the 2D CT images (PNG files) to 3D volume (TIFF file)
for index in range(1, 10):
    image_folder = f'/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset35{index}_Skull/labelsTr'
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.endswith('.png')]

    stack = []
    for file in image_files:
        image = Image.open(file)
        image_array = np.array(image)
        stack.append(image_array)

    volume = np.stack(stack, axis=0)

    output_tiff = f'/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/tifvolume/0{index}_0000.tif' # 0{index}.tif for masks
    tifffile.imwrite(output_tiff, volume, photometric='minisblack')
