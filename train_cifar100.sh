#!/bin/bash

set -euo pipefail

# Change working directory
cd "$HOME/Documents/DeepLipschitzResNet/" || exit 1

# Load conda environment
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate DeepLipschitzResNet

DATASET="cifar100"
MODEL_NAME="medium"
TRAIN_DIR="medium_model"

OUTPUT_DIR="./trained_models/${TRAIN_DIR}/${DATASET}"

# Create output directory if it does not exists
mkdir -p "$OUTPUT_DIR"

#Check that torchrun exists
command -v torchrun >/dev/null || { echo "Error: torchrun not found in PATH"; exit 1; }

NPROC_TRAIN=4
NPROC_CERT=1

for i in {50..50} ; do
    echo "Training with seed $i on $DATASET..."
    torchrun --standalone --nproc_per_node="$NPROC_TRAIN" src/main.py --dataset "$DATASET" --model-name "$MODEL_NAME" --train_dir "$TRAIN_DIR" --optimizer adamw --scheduler cosine --lr 0.001 --activation elu --no-gradient_layer_normalization --no-batch_norm --no-dropout --seed "$i" 2>&1 | tee "${OUTPUT_DIR}/output${i}.txt"

    echo "Running certified mode with seed $i on $DATASET..."
    torchrun --standalone --nproc_per_node="$NPROC_CERT" src/main.py --dataset "$DATASET" --model-name "$MODEL_NAME" --train_dir "$TRAIN_DIR" --optimizer adamw --scheduler cosine --lr 0.001 --no-gradient_layer_normalization --activation elu  --no-batch_norm --no-dropout --mode certified --seed "$i" 2>&1 | tee "${OUTPUT_DIR}/certified_output${i}.txt"
done
