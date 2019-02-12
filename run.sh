#!/bin/bash

# --
# AG news

python prep.py \
    --dataset ag \
    --train-path $HOME/data/fasttext/ag_news_csv/train.csv \
    --valid-path $HOME/data/fasttext/ag_news_csv/test.csv \
    --outpath ./data/ag_news.tsv

python featurize.py \
    --inpath ./data/ag_news.tsv \
    --outdir ./results/ag_news

CUDA_VISIBLE_DEVICES=6 python finetune_lm.py \
    --df-path data/ag_news.tsv \
    --rundir results/ag_news

python shallow_classifier.py \
    --lm-weights-path results/ag_news/lm_weights/lm_ft_final-epoch14.h5 \
    --df-path data/ag_news.tsv

python deep_classifier.py \
    --lm-weights-path results/ag_news/lm_ft_final-epoch14.h5 \
    --df-path data/ag_news.tsv
