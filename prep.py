#!/usr/bin/env python

"""
    prep.py
"""

import os
import pandas as pd
import numpy as np

# --
# Helpers

def join_columns(df, cols, begin_tok='xbos', field_tok='xfld'):
    text = f'\n{begin_tok}'
    for i, c in enumerate(cols):
        text += f' {field_tok} {i} ' + df[c].astype(str)
    
    return text

def prep_ag(df, cl_train):
    df.columns     = ('label', 'headline', 'body')
    df['text']     = join_columns(df, ['headline', 'body'])
    df['cl_train'] = cl_train
    df['lm_train'] = np.random.uniform(0, 1, df.shape[0]) < 0.9
    del df['headline']
    del df['body']
    
    return df

# --
# Run

seed    = 123
dataset = 'ag'

np.random.seed(seed)

train_path = os.path.expanduser('~/data/fasttext/ag_news_csv/train.csv')
valid_path = os.path.expanduser('~/data/fasttext/ag_news_csv/test.csv')
outpath    = './data/ag.tsv'

train_df = pd.read_csv(train_path, header=None)
valid_df = pd.read_csv(valid_path, header=None)

if dataset == 'ag':
    train_df = prep_ag(train_df, cl_train=True)
    valid_df = prep_ag(valid_df, cl_train=False)
else:
    raise Exception

df = pd.concat([train_df, valid_df], axis=0)
df = df.sample(df.shape[0], replace=False)
df = df.reset_index(drop=True)

ulabel = set(df.label)
label_lookup = dict(zip(ulabel, range(len(ulabel))))

df.label = df.label.apply(label_lookup.get)

df = df[['lm_train', 'cl_train', 'label', 'text']]

df.to_csv(outpath, sep='\t', index=False)