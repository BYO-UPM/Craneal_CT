import numpy as np
from skimage import io

def dice_score(image1_path, image2_path):
    # Read the TIFF images
    image1 = io.imread(image1_path)
    image2 = io.imread(image2_path)
    
    # Convert to binary masks (assuming the images are already in binary form)
    binary_mask1 = image1 > 0
    binary_mask2 = image2 > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(binary_mask1, binary_mask2).sum()
    union = binary_mask1.sum() + binary_mask2.sum()
    
    # Calculate Dice score
    dice = 2 * intersection / union
    return dice

image1_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/tifvolume/mask/06.tif'
image2_path = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/result_com_final/fold6/swintr/06.tif'
score = dice_score(image1_path, image2_path)
print(f'Dice Score: {score}')