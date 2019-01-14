#!/usr/bin/env python

"""
    featurize-v2.py
"""

import re
import html
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

from fastai_tokenizer import Tokenizer

# --
# Helpers

def _zero_fn():
    return 0

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'")\
        .replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n")\
        .replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
        .replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return re.sub(r' +', ' ', html.unescape(x))

def build_vocab(tdocs):
    tok_freq = Counter(a for b in tdocs for a in b)
    vocab = [tok for tok,cnt in tok_freq.most_common(max_vocab) if cnt > min_freq]
    vocab.insert(0, '_pad_')
    vocab.insert(0, '_unk_')
    return defaultdict(_zero_fn, {tok:idx for idx,tok in enumerate(vocab)})


# --
# IO

n_jobs    = 32
inpath    = './data/ag.tsv'
max_vocab = 30000
min_freq  = 2
outpath   = './data/ag'

df = pd.read_csv(inpath, sep='\t')

# --
# Tokenize

docs         = df.text.apply(fixup)
chunked_docs = np.array_split(docs.values.astype(str), n_jobs)
tok_docs     = Tokenizer().proc_all_mp(chunked_docs)

# --
# Convert to IDs

lm_train_idx      = np.where(df.lm_train)[0]
tok_docs_lm_train = [tok_docs[idx] for idx in lm_train_idx]
str2id            = build_vocab(tok_docs_lm_train)

id_docs = [[str2id[tok] for tok in doc] for doc in tok_docs]

# --
# Save

np.save('%s.tok.npy' % outpath, tok_docs)
np.save('%s.id.npy' % outpath, id_docs)
pickle.dump(str2id, open('%s.str2id.pkl', 'wb'))

