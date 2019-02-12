#!/usr/bin/env python

"""
    finetune_lm.py
"""

import argparse

import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd

from basenet.helpers import to_numpy, set_seeds, set_freeze
from ulmfit import LanguageModelLoader, LanguageModel, basenet_train

assert torch.__version__.split('.')[1] == '3', 'Downgrade to pytorch==0.3.2 (for now)'

emb_sz  = 400
nhid    = 1150 
nlayers = 3
bptt    = 70
bs      = 52
drops   = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
lrs     = 1e-3
wd      = 1e-7

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str, default='models/wt103/fwd_wt103.h5')
    parser.add_argument('--lm-vocab-path', type=str, default='models/wt103/itos_wt103.pkl')
    parser.add_argument('--df-path',     type=str, default='data/ag.tsv')
    parser.add_argument('--rundir',      type=str, default='results/ag2/')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Helpers

def load_docs(df_path, doc_path, split='lm_train'):
    train_sel = pd.read_csv(df_path, sep='\t', usecols=[split]).values.squeeze()
    docs = np.load(doc_path)
    return docs[train_sel], docs[~train_sel]


def load_lm_weights(lm_weights_path, lm_vocab_path, str2id_path):
    # Load pretrained weights
    lm_weights = torch.load(lm_weights_path, map_location=lambda storage, loc: storage)
    
    # Load pretrained vocab
    lm_vocab  = pickle.load(open(lm_vocab_path, 'rb'))
    lm_str2id = {v:k for k,v in enumerate(lm_vocab)}
    
    # Load dataset vocab
    str2id = pickle.load(open(str2id_path, 'rb'))
    n_tok  = len(str2id)
    
    # Adjust vocabulary to match finetuning corpus
    lm_enc_weights = to_numpy(lm_weights['0.encoder.weight'])
    
    tmp = np.zeros((n_tok, lm_enc_weights.shape[1]), dtype=np.float32)
    tmp += lm_enc_weights.mean(axis=0)
    for tok, idx in str2id.items():
        if tok in lm_str2id:
            tmp[idx] = lm_enc_weights[lm_str2id[tok]]
    
    lm_weights['0.encoder.weight']                    = torch.Tensor(tmp.copy())
    lm_weights['0.encoder_with_dropout.embed.weight'] = torch.Tensor(tmp.copy())
    lm_weights['1.decoder.weight']                    = torch.Tensor(tmp.copy())
    
    return lm_weights, n_tok

# --
# Run

args = parse_args()
set_seeds(args.seed)
os.makedirs(os.path.join(args.rundir, 'lm_weights'), exist_ok=True)

# --
# Load data

X_train, X_valid = load_docs(
    df_path=args.df_path, 
    doc_path=os.path.join(args.rundir, 'id_docs.npy'),
    split='lm_train',
)

dataloaders = {
    "train" : LanguageModelLoader(np.concatenate(X_train), bs=bs, bptt=bptt),
    "valid" : LanguageModelLoader(np.concatenate(X_valid), bs=bs, bptt=bptt),
}

# --
# Define model

print('finetune_lm.py: loading LM weights from %s' % args.lm_weights_path, file=sys.stderr)
lm_weights, n_tok = load_lm_weights(
    lm_weights_path=args.lm_weights_path,
    lm_vocab_path=args.lm_vocab_path,
    str2id_path=os.path.join(args.rundir, 'str2id.pkl'),
)

model = LanguageModel(
    n_tok     = n_tok, 
    emb_sz    = emb_sz,
    nhid      = nhid, 
    nlayers   = nlayers, 
    pad_token = 1, 
    dropouti  = drops[0],
    dropout   = drops[1],
    wdrop     = drops[2],
    dropoute  = drops[3],
    dropouth  = drops[4],
).to('cuda')

model.verbose = True
print('!! loading weights', file=sys.stderr)
model.load_weights(lm_weights)
set_freeze(model, False)
_ = model.train()
torch.save(model.state_dict(), os.path.join(args.rundir, 'lm_weights', 'lm_orig.h5'))

# --
# Finetune decoder

set_freeze(model.encoder.rnns, True)
set_freeze(model.encoder.dropouths, True)
lm_ft_last = basenet_train(
    model,
    dataloaders,
    num_epochs=1,
    lr_breaks=[0, 0.5, 1],
    lr_vals=[lrs / 64, lrs / 2, lrs / 64],
    adam_betas=(0.8, 0.99),
    weight_decay=wd,
    save_prefix=os.path.join(args.rundir, 'lm_weights', 'lm_ft_last'),
)

# --
# Finetune end-to-end

set_freeze(model.encoder, False)
lm_ft_all = basenet_train(
    model,
    dataloaders,
    num_epochs=15,
    lr_breaks=[0, 15 / 10, 15],
    lr_vals=[lrs / 20, lrs, lrs / 20],
    adam_betas=(0.8, 0.99),
    weight_decay=wd,
    save_prefix=os.path.join(args.rundir, 'lm_weights', 'lm_ft_final'),
)