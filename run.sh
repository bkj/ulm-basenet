#!/bin/bash

# Convert dataset to appropriate formate
python prep.py \
    --dataset ag \
    --train-path $HOME/data/fasttext/ag_news_csv/train.csv \
    --valid-path $HOME/data/fasttext/ag_news_csv/test.csv \
    --outpath ./data/ag2.tsv

python featurize.py \
    --inpath ./data/ag2.tsv \
    --outdir ./results/ag2

python finetune_lm.py
python train_classifier.py