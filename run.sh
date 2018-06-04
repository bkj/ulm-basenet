#!/bin/bash


# Feth data
mkdir -p data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz && mv aclImdb data/aclImdb/

# Make sure you have spacy english model downloaded
python -m spacy download en

wget -nH -r -np -P data/ http://files.fast.ai/models/wt103/

# --
# Run

# Train/test split (classifier and language model)
python prep.py \
    --inpath data/aclImdb \
    --outpath data/run3

# Finetune language model
python featurize.py \
    --inpath data/run3/lm/train.csv \
    --outpath data/run3/lm/train/ \
    --save-itos data/run3/itos.pkl
    --no-labels

python featurize.py \
    --inpath data/run3/lm/valid.csv \
    --outpath data/run3/lm/valid/ \
    --load-itos data/run3/itos.pkl \
    --no-labels

python finetune_lm.py \
    --lm-weights-path data/models/wt103/fwd_wt103.h5 \
    --lm-itos-path data/models/wt103/itos_wt103.pkl \
    --itos-path data/run3/lm/train/itos.pkl \
    --outpath data/run3/lm/weights/ \
    --X-train data/run3/lm/train-X.npy \
    --X-valid data/run3/lm/valid-X.npy > data/run3/lm.jl


# Train classifier
python featurize.py \
    --inpath data/run3/classifier/train.csv \
    --outpath data/run3/classifier/train/ \
    --load-itos data/run3/itos.pkl

python featurize.py \
    --inpath data/run3/classifier/valid.csv \
    --outpath data/run3/classifier/valid/ \
    --load-itos data/run3/itos.pkl

python train_classifier.py \
    --lm-weights-path data/run3/lm/weights/lm_ft_final.h5 \
    --outpath data/run3/classifier/weights \
    --X-train data/run3/classifier/train-X.npy \
    --X-valid data/run3/classifier/valid-X.npy \
    --y-train data/run3/classifier/train-y.npy \
    --y-valid data/run3/classifier/valid-y.npy > data/run3/classifier.jl
