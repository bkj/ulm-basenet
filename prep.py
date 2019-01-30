#!/usr/bin/env python

"""
    prep.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd


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


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='ag', choices=['ag'])
    parser.add_argument('--train-path', type=str)
    parser.add_argument('--valid-path', type=str)
    parser.add_argument('--outpath', type=str, default='./data/ag.tsv')
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    np.random.seed(args.seed)
    
    print('prep.py: read from %s' % args.train_path, file=sys.stderr)
    train_df = pd.read_csv(args.train_path, header=None)
    
    print('prep.py: read from %s' % args.valid_path, file=sys.stderr)
    valid_df = pd.read_csv(args.valid_path, header=None)
    
    print('prep.py: preprocess', file=sys.stderr)
    if args.dataset == 'ag':
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
    
    df.to_csv(args.outpath, sep='\t', index=False)
    print('prep.py: wrote to %s' % args.outpath, file=sys.stderr)
