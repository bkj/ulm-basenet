#!/usr/bin/env python

"""
    shallow_classifier.py
"""

import argparse

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

from torch.nn import functional as F
from torch.utils.data import DataLoader

from basenet.text.data import RaggedDataset, text_collate_fn
from basenet.helpers import to_numpy, set_seeds, set_freeze
from ulmfit import TextClassifier, basenet_train

assert torch.__version__.split('.')[1] == '3', 'Downgrade to pytorch==0.3.2 (for now)'

# --
# Params

bptt, emb_sz, n_hid, n_layers, batch_size = 70, 400, 1150, 3, 48
dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
lr  = 3e-3
lrm = 2.6
lrs = np.array([lr / (lrm ** i) for i in range(5)[::-1]])
max_seq = 20 * 70
pad_token = 1

# --
# Helpers

def load_cl_docs(df_path, doc_path, do_sort=True):
    tmp = pd.read_csv(df_path, sep='\t', 
        usecols=['cl_train', 'label'],
        dtype={
            'cl_train' : bool,
            'label'    : int,
        }
    )
    
    train_sel = tmp.cl_train.values
    label     = tmp.label.values
    
    docs = np.load(doc_path)
    X_train, X_valid = docs[train_sel], docs[~train_sel]
    y_train, y_valid = label[train_sel], label[~train_sel]
    
    if do_sort:
        train_ord = np.argsort([-len(x) for x in X_train])
        valid_ord = np.argsort([-len(x) for x in X_valid])
    else:
        train_ord = np.random.permutation(X_train.shape[0])
        valid_ord = np.random.permutation(X_valid.shape[0])
    
    X_train, y_train = X_train[train_ord], y_train[train_ord]
    X_valid, y_valid = X_valid[valid_ord], y_valid[valid_ord]
    
    return X_train, X_valid, y_train, y_valid


# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str, default='results/ag/lm_ft_final-epoch14.h5')
    parser.add_argument('--df-path', type=str, default='data/ag_news.tsv')
    parser.add_argument('--rundir',  type=str, default='results/ag2/')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


args = parse_args()
set_seeds(args.seed)

# --
# IO

X_train, X_valid, y_train, y_valid = load_cl_docs(
    df_path=args.df_path, 
    doc_path=os.path.join(args.rundir, 'id_docs.npy'),
    do_sort=True,
)

dataloaders = {
    "train" : DataLoader(
        dataset=RaggedDataset(X_train, y_train),
        collate_fn=text_collate_fn,
        shuffle=False,
        batch_size=batch_size,
    ),
    "valid" : DataLoader(
        dataset=RaggedDataset(X_valid, y_valid),
        collate_fn=text_collate_fn,
        shuffle=False,
        batch_size=batch_size,
    )
}

# --
# Define model

def text_classifier_loss_fn(x, target, alpha=0, beta=0):
    assert isinstance(x, tuple), 'not isinstance(x, tuple)'
    assert len(x) == 3, 'len(x) != 3'
    
    l_x, last_raw_output, last_output = x
    
    # Cross entropy loss
    loss = F.cross_entropy(l_x, target)
    
    # Activation Regularization
    if alpha:
        loss = loss + sum(alpha * last_output.pow(2).mean())
    
    # Temporal Activation Regularization (slowness)
    if beta: 
        if len(last_raw_output) > 1:
            loss = loss + sum(beta * (last_raw_output[1:] - last_raw_output[:-1]).pow(2).mean())
    
    return loss


lm_weights = torch.load(args.lm_weights_path)

n_tok   = lm_weights['encoder.encoder.weight'].shape[0]
n_class = len(set(y_train))

model = TextClassifier(
    bptt        = bptt,
    max_seq     = max_seq,
    n_class     = n_class,
    n_tok       = n_tok,
    emb_sz      = emb_sz,
    n_hid       = n_hid,
    n_layers    = n_layers,
    pad_token   = pad_token,
    head_layers = [emb_sz * 3, 50, n_class],
    head_drops  = [dps[4], 0.1],
    dropouti    = dps[0],
    wdrop       = dps[1],
    dropoute    = dps[2],
    dropouth    = dps[3],
    loss_fn     = partial(text_classifier_loss_fn, alpha=2, beta=1),
).to('cuda')
model.verbose = True
print(model, file=sys.stderr)

# >>
# !! Should maybe save encoder weights separately in `finetune_lm.py`
weights_to_drop = [k for k in lm_weights.keys() if 'decoder.' in k]
for k in weights_to_drop:
    del lm_weights[k]
# <<

model.load_state_dict(lm_weights, strict=False)
set_freeze(model, False)

# --
# Train

# Finetune decoder
set_freeze(model.encoder.encoder, True)
set_freeze(model.encoder.dropouti, True)
set_freeze(model.encoder.rnns, True)
set_freeze(model.encoder.dropouths, True)

class_ft_dec = basenet_train(
    model,
    dataloaders,
    num_epochs=1,
    lr_breaks=[0, 1/3, 1],
    lr_vals=[lrs / 8, lrs, lrs / 8],
    adam_betas=(0.7, 0.99),
    weight_decay=0,
    clip_grad_norm=25,
    save_prefix=os.path.join(args.rundir, 'cl_ft_last1'),
)

# Finetune last layer
set_freeze(model.encoder.rnns[-1], False)
set_freeze(model.encoder.dropouths[-1], False)
class_ft_last = basenet_train(
    model,
    dataloaders,
    num_epochs=1,
    lr_breaks=[0, 1/3, 1],
    lr_vals=[lrs / 8, lrs, lrs / 8],
    adam_betas=(0.7, 0.99),
    weight_decay=0,
    clip_grad_norm=25,
    save_prefix=os.path.join(args.rundir, 'cl_ft_last2'),
)

# Finetune end-to-end
set_freeze(model, False)
class_ft_all = basenet_train(
    model,
    dataloaders,
    num_epochs=14,
    lr_breaks=[0, 14/10, 14],
    lr_vals=[lrs / 32, lrs, lrs / 32],
    adam_betas=(0.7, 0.99),
    weight_decay=0,
    clip_grad_norm=25,
    save_prefix=os.path.join(args.rundir, 'cl_final'),
)

