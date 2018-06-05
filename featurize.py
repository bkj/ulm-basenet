#!/usr/bin/env python

"""
    featurize.py
"""

import os
import re
import sys
import html
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from fastai_tokenizer import Tokenizer

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=True)
    parser.add_argument('--outpath', type=str, required=True)
    parser.add_argument('--save-itos', type=str)
    parser.add_argument('--load-itos', type=str)
    parser.add_argument('--max-vocab', type=int, default=30000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--no-labels', action="store_true")
    args = parser.parse_args()
    assert not (args.load_itos and args.save_itos), 'only one of `--save-itos` or `--load-itos` may be set'
    assert (args.load_itos or args.save_itos), 'one of `--save-itos` or `--load-itos` must be set'
    return args

# --
# Helpers

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'")\
        .replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n")\
        .replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
        .replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return re.sub(r' +', ' ', html.unescape(x))

def get_texts(df, n_lbls=1, num_cpus=32, BOS='xbos', FLD='xfld'):
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls + 1, len(df.columns)):
        texts += f' {FLD} {i - n_lbls} ' + df[i].astype(str)
    
    texts = texts.apply(fixup).values.astype(str)
    text_chunks = np.array_split(texts, num_cpus)
    tok_texts = Tokenizer().proc_all_mp(text_chunks)
    
    labels = list(df.iloc[:,range(n_lbls)].values.squeeze())
    return tok_texts, labels

def get_all(df, n_lbls):
    tok, labels = [], []
    n_lines = 0
    for df_chunk in df:
        tok_, labels_ = get_texts(df_chunk, n_lbls)
        tok.append(tok_)
        labels.append(labels_)
        
        n_lines += df_chunk.shape[0]
        print('processed %d lines' % n_lines, file=sys.stderr)
    
    return sum(tok, []), sum(labels, [])

# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    print('featurize.py: loading %s' % args.inpath, file=sys.stderr)
    df = pd.read_csv(args.inpath, header=None, chunksize=24000)
    tok, labels = get_all(df, n_lbls=1)
    
    if args.save_itos:
        print('featurize.py: computing itos and writing to %s' % args.save_itos, file=sys.stderr)
        freq = Counter(p for o in tok for p in o)
        itos = [o for o,c in freq.most_common(args.max_vocab) if c > args.min_freq]
        itos.insert(0, '_pad_')
        itos.insert(0, '_unk_')
        pickle.dump(itos, open(args.save_itos, 'wb'))
    else:
        print('featurize.py: loading itos from %s' % args.load_itos, file=sys.stderr)
        itos = pickle.load(open(args.load_itos, 'rb'))
    
    stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    ids  = np.array([[stoi[o] for o in p] for p in tok])
    
    print('featurize.py: writing to %s' % args.outpath, file=sys.stderr)
    np.save('%s-%s' % (args.outpath, 'tok.npy'), tok)
    np.save('%s-%s' % (args.outpath, 'X.npy'), ids)
    if not args.no_labels:
        np.save('%s-%s' % (args.outpath, 'y.npy'), labels)
