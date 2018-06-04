#!/usr/bin/env python

"""
    make-splits.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/aclImdb')
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--seed', type=str, default=123)
    return parser.parse_args()

def get_texts(path):
    texts, labels = [], []
    for label in ['neg', 'pos', 'unsup']:
        for fname in glob(os.path.join(path, label, '*.*')):
            yield open(fname, 'r').read(), label

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    
    os.makedirs(args.outpath, exist_ok=True)
    os.makedirs(os.path.join(args.outpath, 'classifier'), exist_ok=True)
    os.makedirs(os.path.join(args.outpath, 'lm'), exist_ok=True)
    
    # --
    # IO
    
    print('prep.py: reading from %s' % args.inpath, file=sys.stderr)
    train_data = get_texts(os.path.join(args.inpath, 'train'))
    X_train, y_train = zip(*train_data)
    
    val_data = get_texts(os.path.join(args.inpath, 'test'))
    X_valid, y_valid = zip(*val_data)
    
    # --
    # Classifier data
    print('prep.py: splitting classifier data', file=sys.stderr)
    
    # Train
    cl_df_train = pd.DataFrame({"labels" : y_train, "text" : X_train}, columns=['labels', 'text'])
    cl_df_train = cl_df_train.sample(cl_df_train.shape[0], replace=False)
    cl_df_train = cl_df_train[cl_df_train.labels != 'unsup'] # Drop unsupervised
    cl_df_train.to_csv(os.path.join(args.outpath, 'classifier/train.csv'), header=False, index=False)
    
    # Valid
    cl_df_valid = pd.DataFrame({"labels" : y_valid, "text" : X_valid}, columns=['labels', 'text'])
    cl_df_valid = cl_df_valid.sample(cl_df_valid.shape[0], replace=False)
    cl_df_valid.to_csv(os.path.join(args.outpath, 'classifier/valid.csv'), header=False, index=False)
    
    # --
    # LM data
    print('prep.py: splitting language model data', file=sys.stderr)
    
    X_all = np.concatenate([X_train, X_valid])
    lm_train, lm_valid = train_test_split(X_all, test_size=0.1)
    
    lm_df_train = pd.DataFrame({"labels" : 0, "text" : lm_train}, columns=['labels', 'text'])
    lm_df_train.to_csv(os.path.join(args.outpath, 'lm/train.csv'), header=False, index=False)
    
    lm_df_valid = pd.DataFrame({"labels" : 0, "text" : lm_valid}, columns=['labels', 'text'])
    lm_df_valid.to_csv(os.path.join(args.outpath, 'lm/valid.csv'), header=False, index=False)
    
    print('prep.py: wrote to %s' % os.path.join(args.outpath, '{classifier,lm}/{train,valid}'), file=sys.stderr)