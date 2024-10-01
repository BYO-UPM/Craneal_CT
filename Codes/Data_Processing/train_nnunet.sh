#!/bin/bash

# Set the model name here
MODEL_NAME="nnUNetTrainerSwinUNETR"

# Set the CUDA device here
GPU_DEVICE=2

# Loop through the folds 351 to 359
for fold_id in {501..504}; do
  # Set the CUDA device for the training process
  export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
  
  # Execute the training command for the current fold
  nnUNetv2_train $fold_id 2d all -tr $MODEL_NAME
  
  echo "Completed training for fold ${fold_id} using model ${MODEL_NAME} on GPU ${GPU_DEVICE}"
done

echo "All training completed for model ${MODEL_NAME} on GPU ${GPU_DEVICE}."
