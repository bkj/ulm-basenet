#!/usr/bin/env python

"""
    featurize-v2.py
"""

import os
import re
import sys
import html
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from fastai_tokenizer import Tokenizer

# --
# Helpers

def clean_text(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'")\
        .replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n")\
        .replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
        .replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return re.sub(r' +', ' ', html.unescape(x))


def build_vocab(tdocs, max_vocab=30000, min_token_freq=2):
    tok_freq = Counter(a for b in tdocs for a in b)
    vocab = [tok for tok,cnt in tok_freq.most_common(max_vocab) if cnt > min_token_freq]
    vocab.insert(0, '_pad_')
    vocab.insert(0, '_unk_')
    return {tok:idx for idx,tok in enumerate(vocab)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./data/ag.tsv')
    parser.add_argument('--outdir', type=str, default='./results/ag')
    parser.add_argument('--n-jobs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--max-vocab', type=int, default=30000)
    parser.add_argument('--min-token-freq', type=int, default=2)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print('reading %s' % args.inpath, file=sys.stderr)
    if '.tsv' in args.inpath:
        df = pd.read_csv(args.inpath, sep='\t')
    elif '.feather' in args.inpath:
        df = pd.read_feather(args.inpath)
    else:
        raise Exception
    
    # --
    # Tokenize
    
    print('tokenizing', file=sys.stderr)
    
    docs         = df.text.apply(clean_text)
    tok_docs     = Tokenizer().proc_all_mp(docs.values.astype(str), n_jobs=args.n_jobs)
    
    # --
    # Convert to IDs
    
    print('convert to ints', file=sys.stderr)
    
    lm_train_idx      = np.where(df.lm_train)[0]
    tok_docs_lm_train = [tok_docs[idx] for idx in lm_train_idx]
    
    str2id = build_vocab(
        tdocs=tok_docs_lm_train,
        max_vocab=args.max_vocab, 
        min_token_freq=args.min_token_freq
    )
    
    id_docs = [[str2id.get(tok, 0) for tok in doc] for doc in tok_docs]
    
    # --
    # Save
    
    print('saving to %s/{tok_docs.npy,id_docs.npy,str2id.pkl}' % args.outdir, file=sys.stderr)
    
    np.save(os.path.join(args.outdir, 'tok_docs.npy'), tok_docs)
    np.save(os.path.join(args.outdir, 'id_docs.npy'), id_docs)
    pickle.dump(str2id, open(os.path.join(args.outdir, 'str2id.pkl'), 'wb'))

