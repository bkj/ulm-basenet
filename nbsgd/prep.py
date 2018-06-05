#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    nbsgd.py
"""

import os
import re
import sys
import string
import argparse
import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


# --
# Helpers

def texts_from_folders(src, names):
    texts, labels = [], []
    for idx, name in enumerate(names):
        path = os.path.join(src, name)
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(open(fpath).read())
            labels.append(idx)
    
    return texts,np.array(labels)


def bow2adjlist(X, maxcols=None):
    x = coo_matrix(X)
    _, counts = np.unique(x.row, return_counts=True)
    pos = np.hstack([np.arange(c) for c in counts])
    adjlist = csr_matrix((x.col + 1, (x.row, pos)))
    datlist = csr_matrix((x.data, (x.row, pos)))
    
    if maxcols is not None:
        adjlist, datlist = adjlist[:,:maxcols], datlist[:,:maxcols]
    
    return adjlist, datlist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-features', type=int, default=200000)
    parser.add_argument('--max-words', type=int, default=1000)
    parser.add_argument('--ngram-range', type=str, default='1,3')
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    
    # --
    # IO
    print("prep.py: loading", file=sys.stderr)
    
    # >>
    # text_train, y_train = texts_from_folders('data/aclImdb/train', ['neg', 'pos'])
    # text_test, y_test = texts_from_folders('data/aclImdb/test', ['neg', 'pos'])
    # --
    train = pd.read_csv('../runs/0/classifier/train.csv', header=None, sep=',')
    y_train, text_train = train.values.T
    
    test = pd.read_csv('../runs/0/classifier/valid.csv', header=None, sep=',')
    y_test, text_test = test.values.T
    
    lm = pd.read_csv('../runs/0/lm/train.csv', header=None, sep=',')
    text_lm = lm.values[:,1]
    y_lm = (np.zeros(text_lm.shape[0]) - 1).astype(int)
    # <<
    
    # --
    # Preprocess
    print("prep.py: preprocessing", file=sys.stderr)
    
    re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)
    tokenizer = lambda x: re_tok.sub(r' \1 ', x).split()
    
    vectorizer = CountVectorizer(
        ngram_range=tuple(map(int, args.ngram_range.split(','))),
        tokenizer=tokenizer, 
        max_features=args.max_features
    )
    X_train = vectorizer.fit_transform(text_train)
    X_test  = vectorizer.transform(text_test)
    X_lm    = vectorizer.transform(text_lm)
    
    X_train_words, _ = bow2adjlist(X_train, maxcols=args.max_words)
    X_test_words, _  = bow2adjlist(X_test, maxcols=args.max_words)
    X_lm_words, _    = bow2adjlist(X_lm, maxcols=args.max_words)
    
    # --
    # Save
    print("prep.py: saving", file=sys.stderr)
    
    np.save('./data/aclImdb/X_train', X_train)
    np.save('./data/aclImdb/X_test', X_test)
    np.save('./data/aclImdb/X_lm', X_test)
    
    np.save('./data/aclImdb/X_train_words', X_train_words)
    np.save('./data/aclImdb/X_test_words', X_test_words)
    np.save('./data/aclImdb/X_lm_words', X_lm_words)
    
    np.save('./data/aclImdb/y_train', y_train)
    np.save('./data/aclImdb/y_test', y_test)
    np.save('./data/aclImdb/y_lm', y_lm)

