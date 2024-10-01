#!/bin/bash

# # Variables you can modify
SOURCE_DATASET="Dataset504_Skull"
# DESTINATION_DATASET="Dataset504_Skull"
# IMAGES_DIR="imagesTr"

# # Base path (modify if your base path is different)
# BASE_PATH="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw"

# # Construct full source and destination paths
# SOURCE_PATH="$BASE_PATH/$SOURCE_DATASET/$IMAGES_DIR"
# DESTINATION_PATH="$BASE_PATH/$DESTINATION_DATASET/$IMAGES_DIR"

# # Create the destination directory if it doesn't exist
# mkdir -p "$DESTINATION_PATH"

# # Copy and rename files
# for filepath in "$SOURCE_PATH"/*; do
#     filename=$(basename "$filepath")
#     extension="${filename##*.}"
#     name="${filename%.*}"
#     new_filename="${name}_true.${extension}"
#     cp "$filepath" "$DESTINATION_PATH/$new_filename"
# done

# echo "All images have been copied from $SOURCE_PATH to $DESTINATION_PATH with '_true' appended to their names."


LABELS_FOLDER="labelsTr"
IMAGES_FOLDER="imagesTr"

# # Base path (modify if your base path is different)
BASE_PATH="/home/ysun@gaps_domain.ssr.upm.es/Craneal_CT/U-Mamba/data/nnUNet_raw"

# # Construct full source and destination paths
# SOURCE_PATH="$BASE_PATH/$SOURCE_DATASET/$IMAGES_DIR"
# DESTINATION_PATH="$BASE_PATH/$DESTINATION_DATASET/$IMAGES_DIR"

# # Create the destination directory if it doesn't exist
# mkdir -p "$DESTINATION_PATH"

# # Copy and rename files
# for filepath in "$SOURCE_PATH"/*; do
#     filename=$(basename "$filepath")
#     extension="${filename##*.}"
#     name="${filename%.*}"
#     new_filename="${name}_true.${extension}"
#     cp "$filepath" "$DESTINATION_PATH/$new_filename"
# done

# echo "All images have been copied from $SOURCE_PATH to $DESTINATION_PATH with '_true' appended to their names."




# Variables you can modify
IMAGES_DIR="$BASE_PATH/$SOURCE_DATASET/$IMAGES_FOLDER"
LABELS_DIR="$BASE_PATH/$SOURCE_DATASET/$LABELS_FOLDER"
# Temporary directory to store renamed files
TEMP_DIR="$(mktemp -d)"

# Function to extract the first two parts (XX and YY) of a filename
extract_key() {
    local filename="$1"
    local base="${filename##*/}"      # Remove path
    base="${base%.*}"                 # Remove extension
    IFS='_' read -r -a parts <<< "$base"
    if [ ${#parts[@]} -ge 2 ]; then
        echo "${parts[0]}_${parts[1]}"
    else
        echo ""
    fi
}

# Create associative arrays to hold file paths
declare -A images_map
declare -A labels_map

# Process images
for img_file in "$IMAGES_DIR"/*; do
    if [ -f "$img_file" ]; then
        key=$(extract_key "$img_file")
        if [ -n "$key" ]; then
            images_map["$key"]="$img_file"
        fi
    fi
done

# Process labels
for lbl_file in "$LABELS_DIR"/*; do
    if [ -f "$lbl_file" ]; then
        key=$(extract_key "$lbl_file")
        if [ -n "$key" ]; then
            labels_map["$key"]="$lbl_file"
        fi
    fi
done

# Rename and match files
for key in "${!images_map[@]}"; do
    if [ -n "${labels_map[$key]}" ]; then
        img_file="${images_map[$key]}"
        lbl_file="${labels_map[$key]}"

        # Get extensions
        extension_img=".png"  # Images should have .png extension
        extension_lbl="${lbl_file##*.}"

        # Construct new filenames
        new_img_name="${key}_0000${extension_img}"
        new_lbl_name="${key}.${extension_lbl}"

        # Rename image
        mv "$img_file" "$IMAGES_DIR/$new_img_name"

        # Rename label
        mv "$lbl_file" "$LABELS_DIR/$new_lbl_name"

        echo "Matched and renamed: Image - $new_img_name, Label - $new_lbl_name"
    else
        echo "No matching label for image key: $key"
    fi
done

# Report unmatched labels
for key in "${!labels_map[@]}"; do
    if [ -z "${images_map[$key]}" ]; then
        echo "No matching image for label key: $key"
    fi
done

echo "Renaming and synchronization complete."