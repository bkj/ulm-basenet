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

from torch.utils.data import DataLoader

from basenet.text.data import RaggedDataset, text_collate_fn
from basenet.helpers import to_numpy, set_seeds, set_freeze
from ulmfit import LanguageModel

assert torch.__version__.split('.')[1] == '3', 'Downgrade to pytorch==0.3.2 (for now)'

# --
# Params

emb_sz  = 400
nhid    = 1150 
nlayers = 3
bptt    = 70
bs      = 52
drops   = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
lrs     = 1e-3
wd      = 1e-7

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
        X_train, y_train = X_train[train_ord], y_train[train_ord]
        
        valid_ord = np.argsort([-len(x) for x in X_valid])
        X_valid, y_valid = X_valid[valid_ord], y_valid[valid_ord]
    
    return X_train, X_valid, y_train, y_valid

def extract_features(model, dataloaders, mode='train'):
    all_feats, all_targets = [], []
    for x, _ in tqdm(dataloaders[mode], total=len(dataloaders[mode])):
        last_output = model.encoder(x.cuda())[1][-1]
        feat = torch.cat([
            last_output[-1],
            last_output.max(dim=0)[0],
            last_output.mean(dim=0)
        ], 1)
        all_feats.append(to_numpy(feat))
        
    return np.vstack(all_feats)

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str, default='results/ag/lm_ft_final-epoch13.h5')
    parser.add_argument('--df-path', type=str, default='data/ag.tsv')
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
)

dataloaders = {
    "train" : DataLoader(
        dataset=RaggedDataset(X_train, y_train),
        collate_fn=text_collate_fn,
        shuffle=False,
        batch_size=32,
    ),
    "valid" : DataLoader(
        dataset=RaggedDataset(X_valid, y_valid),
        collate_fn=text_collate_fn,
        shuffle=False,
        batch_size=32,
    )
}

# --
# Load model

lm_weights = torch.load(args.lm_weights_path)

model = LanguageModel(
    n_tok     = lm_weights['encoder.encoder.weight'].shape[0], 
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
model.load_weights(lm_weights)

_ = model.reset()
set_freeze(model, True)
_ = model.eval()
model.verbose = True
model.use_decoder = False

# --
# Extract features

train_feats = extract_features(model, dataloaders, mode='train')
valid_feats = extract_features(model, dataloaders, mode='valid')






