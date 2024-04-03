import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your mask image - this should be a binary image where the object's pixels are white, and the background is black
# For demonstration, let's create a binary image using NumPy
# Imagine this is your mask
mask_p = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed/S090899/Mask/train2_mask_11.png'
mask_image = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

# Find contours
contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image for drawing contours
contour_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored drawing

# Draw contours
cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 1)  # Drawing in yellow

height, width = mask_image.shape
transparent_image = np.zeros((height, width, 4), dtype=np.uint8)

cv2.drawContours(transparent_image, contours, -1, (0, 255, 255, 255), 1)

cv2.imwrite('contours_on_transparent_background.png', transparent_image)
