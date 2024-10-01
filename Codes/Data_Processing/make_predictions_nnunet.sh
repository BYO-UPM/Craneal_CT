#!/bin/bash

# Set the model name here
MODEL_NAME="nnUNetTrainerSwinUNETR"

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=0

fold_id=504

# Loop through the folds 351 to 359
# for fold_id in {501}; do
# Set the paths dynamically based on the fold_id
input_path="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset${fold_id}_Skull/imagesTs"
# input_path="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset501_Skull/imagesTr"
output_path="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_results/Dataset${fold_id}_Skull/${MODEL_NAME}__nnUNetPlans__2d/predictions"
# output_path="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw/Dataset501_Skull/labelsTr"

# Execute the prediction command
# nnUNetv2_predict -i "$input_path" -o "$output_path" -d $fold_id -c 2d -f all -tr $MODEL_NAME --disable_tta
nnUNetv2_predict -i "$input_path" -o "$output_path" -d 364 -c 2d -f all -tr $MODEL_NAME --disable_tta

echo "Completed predictions for fold ${fold_id} using model ${MODEL_NAME}"
# done

echo "All predictions completed for model ${MODEL_NAME}."