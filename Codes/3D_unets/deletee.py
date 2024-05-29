import sys
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from patchify import unpatchify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import segmentation_models_3D as sm
import cc3d
from scipy.ndimage import binary_fill_holes

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

path_test = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/DL_Models/Model_3D/3D_data_cv/cv"
path_model = "/media/my_ftp/BasesDeDatos_Paranasal_CAT/CT_Craneal/DL_Models/Model_3D/"
encoder = 'vgg16'
#'vanilla', 

scaler = MinMaxScaler()

mean_dice_ori = []
mean_dice_post = []

for i in range(1, 10):
    test_original_3D = np.load(f'{path_test}{i}/3D_test_original{i}.npy')
    test_mask_3D = np.load(f'{path_test}{i}/3D_test_mask{i}.npy')
    model = f'{path_model}{encoder}_3D_cv{i}.h5'

    # Back to 512x512x64
    original_patches_reshaped = np.reshape(test_original_3D[:,:,:,:,0],
                                                (4, 4, 1, 128, 128, 64) )
    mask_patches_reshaped = np.reshape(test_mask_3D[:,:,:,:,0],
                                                (4, 4, 1, 128, 128, 64) )
    reconstructed_original = unpatchify(original_patches_reshaped, (512,512,64))
    reconstructed_mask = unpatchify(mask_patches_reshaped, (512,512,64))

        #if encoder == 'vanilla':
            #test_original_ori = scaler.fit_transform(test_original_3D.reshape(-1, test_original_3D.shape[-1])).reshape(test_original_3D.shape)
            #model3D = load_model(model, compile=False)
            #prediction = model3D.predict(test_original_ori[:,:,:,:,0])
    
        #if encoder == 'resnet50' or encoder == 'vgg16':
    
    preprocess = sm.get_preprocessing(encoder)
    test_original = preprocess(test_original_3D)
    model3D = load_model(model, compile=False)
    prediction = model3D.predict(test_original)
        
    binary = (prediction > 0.5).astype(int)
    ori_patches_reshaped = np.reshape(binary[:,:,:,:,0],
                                        (4, 4, 1, 128, 128, 64))
    reconstructed = unpatchify(ori_patches_reshaped, (512,512,64))

    post_dust = cc3d.dust(
            reconstructed, threshold=500, connectivity=26, in_place=False
        )
    post_clean_holes = binary_fill_holes(post_dust)

    DICE_nopost = dice_coefficient(reconstructed_mask, reconstructed)
    mean_dice_ori.append(DICE_nopost)
    DICE_dust = dice_coefficient(reconstructed_mask, post_dust)
    DICE_hole = dice_coefficient(reconstructed_mask, post_clean_holes)
    mean_dice_post.append(DICE_hole)
    print(f"CV{i}: DICE of using {encoder} encoder without post-processing:", DICE_nopost)
    print(f"       DICE of using {encoder} encoder post-processing (dust):", DICE_dust)
    print(f"       DICE of using {encoder} encoder post-processing (fill holes):", DICE_hole)

print(f"Mean DICE of using {encoder} encoder without post-processing:", np.mean(mean_dice_ori))
print("no, std", np.std(mean_dice_ori))
print(f"Mean DICE of using {encoder} encoder post-processing:", np.mean(mean_dice_post))
print("post, std", np.std(mean_dice_post))