#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 source_directory destination_directory"
    exit 1
fi

source_dir="$1"
destination_dir="$2"

mkdir -p "$destination_dir"
mkdir -p "$destination_dir/train"
mkdir -p "$destination_dir/inference"

patients=($(ls -d "$source_dir"/*/))
train_count=25

for ((i=0; i<$train_count; i++)); do
    cp -r "${patients[$i]}"/* "$destination_dir/train/"
done

for ((i=$train_count; i<${#patients[@]}; i++)); do
    cp -r "${patients[$i]}"/* "$destination_dir/inference/"
done

echo "Dataset split complete"
