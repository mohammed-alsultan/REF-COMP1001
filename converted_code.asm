#!/bin/bash

# Compile the C program
gcc -o image_processor q3.c -lm

# Run the program with provided arguments
# Usage: ./run_image_processor.sh input_image output_blur_image output_edge_image

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_image> <output_blur_image> <output_edge_image>"
    exit 1
fi

./image_processor "$1" "$2" "$3"
