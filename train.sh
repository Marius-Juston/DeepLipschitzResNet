#!/bin/bash

set -e

cd /home/mjuston2/Documents/DeepLipschitzResNet/ || exit

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DeepLipschitzResNet

for i in {42..44} ; do
    torchrun --standalone --nproc_per_node=4 src/main.py --dataset cifar10 --model-name small --train_dir small_model --seed "$i" 2>&1 | tee "/home/mjuston2/Documents/DeepLipschitzResNet/trained_models/small_model/cifar10/output${i}.txt"
done