import cv2
import matplotlib.pyplot as plt
import os

path = "/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/CAT_scans_Preprocessed/S090967/Original"
for file in os.listdir(path):
    if file.endswith(".png"):
        image_path = os.path.join(path, file)
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_image = clahe.apply(image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original CT Image")
    ax[1].imshow(equalized_image, cmap="gray")
    ax[1].set_title("Equalized CT Image")
    plt.show()

