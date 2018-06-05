#!/bin/bash

# run.sh

mkdir -p data
mkdir -p preds

# --
# Train ULM model

# .. train IMDB model using `run.sh` in $PROJECT_ROOT ..

# --
# Predict on training set + unlabeled set

python inference.py \
    --lm-weights-path ../$RUN_PATH/classifier/weights/cl_final-epoch13.h5 \
    --X ../$RUN_PATH/classifier/train-X.npy \
    --outpath preds/classifier-train

python inference.py \
    --lm-weights-path ../$RUN_PATH/classifier/weights/cl_final-epoch13.h5 \
    --X ../$RUN_PATH/lm/train-X.npy \
    --outpath nbsgd/preds/lm-train

# --
# Train NBSGD model (adapted from `https://github.com/bkj/basenet/tree/master/examples/nbsgd`)

cd nbsgd
python prep.py --inpath ../$RUN_PATH
python nbsgd.py 