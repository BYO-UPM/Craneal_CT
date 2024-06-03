from skimage.io import imread
from skimage.filters import threshold_otsu
import numpy as np
import os
import re

# DICE
def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

# Extract last number from a string
def extract_last_number(filename):
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

path_groundtruth = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed/S090981/Mask'
path_prediction = '/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/result_e3_AUFL_DA_cv/cv9'

# Order the files
pred_mask = os.listdir(path_prediction)
pred_mask = sorted(pred_mask, key=extract_last_number)
gt_mask = os.listdir(path_groundtruth)
gt_mask = sorted(gt_mask, key=extract_last_number)
pred_mask = [file for file in pred_mask if file.endswith('.png')]
gt_mask = [file for file in gt_mask if file.endswith('.png')]

dice_coefficients = []

for pred_file, gt_file in zip(pred_mask, gt_mask):
    img_path = os.path.join(path_prediction, pred_file)
    gt_path = os.path.join(path_groundtruth, gt_file)

    gray_image1 = imread(img_path, as_gray=True)
    gray_image2 = imread(gt_path, as_gray=True)

    threshold1 = threshold_otsu(gray_image1)
    threshold2 = threshold_otsu(gray_image2)

    binary_image1 = gray_image1 > threshold1
    binary_image2 = gray_image2 > threshold2

    dice = dice_coefficient(binary_image1,binary_image2)
    dice_coefficients.append(dice)
    #print(dice)

avg_dice_coefficient = np.mean(dice_coefficients)
std_dice_coefficient = np.std(dice_coefficients)
print("Average DICE score:", avg_dice_coefficient)
print("STD DICE score:", std_dice_coefficient)
