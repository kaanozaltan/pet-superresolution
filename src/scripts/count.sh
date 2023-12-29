#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 directory"
    exit 1
fi

target_dir="$1"

if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' not found."
    exit 1
fi

for dir in "$target_dir"/*/; do
    dir_name=$(basename "$dir")
    item_count=$(find "$dir" -maxdepth 1 -type f | wc -l)
    echo "Directory: $dir_name, Item count: $item_count"
done
