#!/bin/bash

set -euo pipefail

# Change working directory
cd "$HOME/Documents/DeepLipschitzResNet/" || exit 1

# Load conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate DeepLipschitzResNet

# Dataset argument (default = cifar10)
DATASET="${1:-cifar10}"

# Validate dataset
case "$DATASET" in
    cifar10|cifar100|tiny-imagenet)
        echo "Using dataset: $DATASET"
        ;;
    *)
        echo "Error: Unsupported dataset '$DATASET'"
        echo "Available datasets: cifar10, cifar100, tiny-imagenet"
        exit 1
        ;;
esac

MODEL_NAME="small"
TRAIN_DIR="small_model"

OUTPUT_DIR="./trained_models/${TRAIN_DIR}/${DATASET}"

# Create output directory if it does not exists
mkdir -p "$OUTPUT_DIR"

#Check that torchrun exists
command -v torchrun >/dev/null || { echo "Error: torchrun not found in PATH"; exit 1; }

NPROC_TRAIN=4
NPROC_CERT=1

for i in {42..44} ; do
    echo "Training with seed $i on $DATASET..."
    torchrun --standalone --nproc_per_node="$NPROC_TRAIN" src/main.py --dataset "$DATASET" --model-name "$MODEL_NAME" --train_dir "$TRAIN_DIR" --seed "$i" 2>&1 | tee "${OUTPUT_DIR}/output${i}.txt"

    echo "Running certified mode with seed $i on $DATASET..."
    torchrun --standalone --nproc_per_node="$NPROC_CERT" src/main.py --dataset "$DATASET" --model-name "$MODEL_NAME" --train_dir "$TRAIN_DIR" --mode certified --seed "$i" 2>&1 | tee "${OUTPUT_DIR}/certified_output${i}.txt"
done