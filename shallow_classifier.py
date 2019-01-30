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
        valid_ord = np.argsort([-len(x) for x in X_valid])
    else:
        train_ord = np.random.permutation(X_train.shape[0])
        valid_ord = np.random.permutation(X_valid.shape[0])
    
    X_train, y_train = X_train[train_ord], y_train[train_ord]
    X_valid, y_valid = X_valid[valid_ord], y_valid[valid_ord]
    
    return X_train, X_valid, y_train, y_valid


def extract_features(model, dataloaders, mode='train'):
    all_feats, all_targets = [], []
    for x, _ in tqdm(dataloaders[mode], total=len(dataloaders[mode])):
        model.reset() # !! Not sure if this is correct.  May make more of a difference for twitter.
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
    do_sort=False,
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

# --
# Train shallow model

from rsub import *
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

X_train_str = [' '.join([str(xxx) for xxx in xx]) for xx in X_train]
X_valid_str = [' '.join([str(xxx) for xxx in xx]) for xx in X_valid]

accs = []
train_sizes = [10, 20, 40, 80, 100, 200, 400, 800, 1600, 3200, 6400]
for train_size in train_sizes:
    
    train_sel = np.random.choice(train_feats.shape[0], train_size, replace=False)
    
    # --
    # ulm
    
    model      = LinearSVC(C=1).fit(train_feats[train_sel], y_train[train_sel])
    pred_valid = model.predict(valid_feats)
    acc_ulm_1  = (y_valid == pred_valid).mean()
    
    model      = LinearSVC(C=0.1).fit(train_feats[train_sel], y_train[train_sel])
    pred_valid = model.predict(valid_feats)
    acc_ulm_01  = (y_valid == pred_valid).mean()
    
    model      = LinearSVC(C=0.001).fit(train_feats[train_sel], y_train[train_sel])
    pred_valid = model.predict(valid_feats)
    acc_ulm_001  = (y_valid == pred_valid).mean()
    
    # --
    # bow
    
    vect     = TfidfVectorizer()
    Xv_train = vect.fit_transform(np.array(X_train_str)[train_sel])
    Xv_valid = vect.transform(X_valid_str)
    
    model      = LinearSVC().fit(Xv_train, y_train[train_sel])
    pred_valid = model.predict(Xv_valid)
    acc_bow    = (y_valid == pred_valid).mean()
    
    accs.append({
        "train_size" : train_size,
        "ulm_1"      : acc_ulm_1,
        "ulm_01"     : acc_ulm_01,
        "ulm_001"    : acc_ulm_001,
        "bow"        : acc_bow,
    })
    print(accs[-1])

accs = pd.DataFrame(accs)

_ = plt.plot(train_sizes, accs.ulm_1, marker='o')
_ = plt.plot(train_sizes, accs.ulm_01, marker='o')
_ = plt.plot(train_sizes, accs.ulm_001, marker='o')
_ = plt.plot(train_sizes, accs.bow, marker='+')
_ = plt.xscale('log')
show_plot()

# --
# Train BOW model
