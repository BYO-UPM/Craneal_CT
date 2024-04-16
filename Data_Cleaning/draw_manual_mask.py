import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_last_number(filename):
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

# Input path
path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/P16/Auto_masks"
filenames = [f for f in sorted(os.listdir(path)) if f.endswith('.png')]
filenames = sorted(filenames, key=extract_last_number)

# Ouput path
output_path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/P16/Auto_lines"

inx_s=0
for f in filenames:
    auto_mask = os.path.join(path, f)
    mask_image = cv2.imread(auto_mask, cv2.IMREAD_GRAYSCALE)

    # Find contours
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a transparent image to draw the 1 pixel contour line
    height, width = mask_image.shape
    transparent_image = np.zeros((height, width, 4), dtype=np.uint8)
    cv2.drawContours(transparent_image, contours, -1, (0, 255, 255, 255), 1)

    # Save the lines
    name = f"P16_auto_lines_{inx_s}.png"
    inx_s = inx_s+1
    output_name = os.path.join(output_path, name)
    cv2.imwrite(output_name, transparent_image)

