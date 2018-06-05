#!/bin/bash

# --
# Setup

# Fetch data
mkdir -p {data,models}
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz && mv aclImdb data/aclImdb/
rm aclImdb_v1.tar.gz

wget http://files.fast.ai/models/wt103/fwd_wt103.h5
wget http://files.fast.ai/models/wt103/itos_wt103.pkl
mkdir models/wt103
mv fwd_wt103.h5 itos_wt103.pkl models/wt103/

# --
# Run

# RUN_PATH="runs/0"
# MAX_VOCAB=30000

RUN_PATH="runs/1"
MAX_VOCAB=15000

# Train/test split (language model and classifier)
python make-splits.py \
    --inpath data/aclImdb \
    --outpath $RUN_PATH

# Featurize datasets (language model and classifier)
python featurize.py \
    --inpath $RUN_PATH/lm/train.csv \
    --outpath $RUN_PATH/lm/train \
    --save-itos $RUN_PATH/itos.pkl \
    --no-labels \
    --max-vocab $MAX_VOCAB

python featurize.py \
    --inpath $RUN_PATH/lm/valid.csv \
    --outpath $RUN_PATH/lm/valid \
    --load-itos $RUN_PATH/itos.pkl \
    --no-labels \
    --max-vocab $MAX_VOCAB

python featurize.py \
    --inpath $RUN_PATH/classifier/train.csv \
    --outpath $RUN_PATH/classifier/train \
    --load-itos $RUN_PATH/itos.pkl \
    --max-vocab $MAX_VOCAB

python featurize.py \
    --inpath $RUN_PATH/classifier/valid.csv \
    --outpath $RUN_PATH/classifier/valid \
    --load-itos $RUN_PATH/itos.pkl \
    --max-vocab $MAX_VOCAB

# Finetune LM
python finetune_lm.py \
    --lm-weights-path models/wt103/fwd_wt103.h5 \
    --lm-itos-path models/wt103/itos_wt103.pkl \
    --itos-path $RUN_PATH/itos.pkl \
    --outpath $RUN_PATH/lm/weights/ \
    --X-train $RUN_PATH/lm/train-X.npy \
    --X-valid $RUN_PATH/lm/valid-X.npy | tee $RUN_PATH/lm.jl

# Train classifier
python train_classifier.py \
    --lm-weights-path $RUN_PATH/lm/weights/lm_ft_final-epoch14.h5 \
    --outpath $RUN_PATH/classifier/weights \
    --X-train $RUN_PATH/classifier/train-X.npy \
    --X-valid $RUN_PATH/classifier/valid-X.npy \
    --y-train $RUN_PATH/classifier/train-y.npy \
    --y-valid $RUN_PATH/classifier/valid-y.npy | tee $RUN_PATH/classifier.jl


# Perform inference
mkdir -p $RUN_PATH/classifier/preds/
python inference.py \
    --lm-weights-path $RUN_PATH/classifier/weights/cl_final-epoch13.h5 \
    --X $RUN_PATH/classifier/train-X.npy \
    --outpath $RUN_PATH/classifier/preds/classifier-train

python inference.py \
    --lm-weights-path $RUN_PATH/classifier/weights/cl_final-epoch13.h5 \
    --X $RUN_PATH/lm/train-X.npy \
    --outpath $RUN_PATH/classifier/preds/lm-train
