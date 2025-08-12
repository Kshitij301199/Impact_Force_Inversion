#!/bin/bash

# Define log directories
out_dir="./logs/out/"
err_dir="./logs/err/"

# Remove all .txt files in the specified directories
rm -v "$out_dir"*.txt "$err_dir"*.txt

echo "Cleanup complete."