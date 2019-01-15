#!/bin/bash

# --
# AG news

# Convert dataset to appropriate format

python prep.py \
    --dataset ag \
    --train-path $HOME/data/fasttext/ag_news_csv/train.csv \
    --valid-path $HOME/data/fasttext/ag_news_csv/test.csv \
    --outpath ./data/ag2.tsv

python featurize.py \
    --inpath ./data/ag2.tsv \
    --outdir ./results/ag2

python finetune_lm.py \
    --df-path data/ag2.tsv \
    --rundir results/ag2

# step through this
# python shallow_classifier.py \
#     --lm-weights-path results/ag/lm_ft_final-epoch13.h5 \
#     --df-path data/ag2.tsv

# --

function pluck_english {
    cat | jq -c 'if(.lang == "en") then {text} else null end //empty' | fgrep "#"
}
export -f pluck_english

cat jon.jl | parallel -N 100000 --pipe pluck_english | gzip -c > jon-en.jl.gz