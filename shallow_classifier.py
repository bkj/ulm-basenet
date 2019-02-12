#!/usr/bin/env python

"""
    shallow_classifier.py
"""

from rsub import *
from matplotlib import pyplot as plt

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.utils.data import DataLoader

from basenet.text.data import RaggedDataset, text_collate_fn
from basenet.helpers import to_numpy, set_seeds, set_freeze

from ulmfit import LanguageModel
from helpers import load_cl_docs, extract_features

assert torch.__version__.split('.')[1] == '3', 'Downgrade to pytorch==0.3.2 (for now)'

# --
# Params

emb_sz  = 400
nhid    = 1150 
nlayers = 3
# bptt    = 70
# bs      = 52
drops   = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.7
# lrs     = 1e-3
# wd      = 1e-7

batch_size = 32

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm-weights-path', type=str, default='results/ag_news/lm_weights/lm_ft_final-epoch13.h5')
    parser.add_argument('--df-path', type=str, default='data/ag_news.tsv')
    parser.add_argument('--rundir',  type=str, default='results/ag_news/')
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

metric = 'accuracy'

def ulm_score(train_feats, valid_feats, y_train, y_valid, C):
    model = LinearSVC(C=C).fit(train_feats, y_train)
    if metric == 'accuracy':
        pred_valid = model.predict(valid_feats)
        return metrics.accuracy_score(y_valid, pred_valid)
    else:
        raise Exception

def bow_score(Xv_train, Xv_valid, y_train, y_valid, C):
    model = LinearSVC(C=C).fit(Xv_train, sub_y_train)
    
    if metric == 'accuracy':
        pred_valid = model.predict(Xv_valid)
        return metrics.accuracy_score(y_valid, pred_valid)
    else:
        raise Exception


X_train_str = np.array([' '.join([str(xxx) for xxx in xx]) for xx in X_train])
X_valid_str = np.array([' '.join([str(xxx) for xxx in xx]) for xx in X_valid])

np.random.seed(args.seed + 111)

hist = []
train_sizes = [5, 10, 20, 40, 80, 100, 200, 400, 800, 1600, 3200]
for train_size in train_sizes:
    
    train_idx, _ = train_test_split(
        np.arange(train_feats.shape[0]), 
        train_size=train_size, 
        stratify=y_train
    )
    
    sub_train_feats = train_feats[train_idx]
    sub_y_train     = y_train[train_idx]
    sub_X_train_str = X_train_str[train_idx]
    
    # --
    # ulm
    
    ulm_score_1 = ulm_score(sub_train_feats, valid_feats, sub_y_train, y_valid, C=1)
    ulm_score_2 = ulm_score(sub_train_feats, valid_feats, sub_y_train, y_valid, C=0.1)
    ulm_score_3 = ulm_score(sub_train_feats, valid_feats, sub_y_train, y_valid, C=0.01)
    
    # --
    # bow
    
    vect     = TfidfVectorizer()
    Xv_train = vect.fit_transform(sub_X_train_str)
    Xv_valid = vect.transform(X_valid_str)
    
    bow_score_1 = bow_score(Xv_train, Xv_valid, sub_y_train, y_valid, C=1)
    bow_score_2 = bow_score(Xv_train, Xv_valid, sub_y_train, y_valid, C=0.1)
    bow_score_3 = bow_score(Xv_train, Xv_valid, sub_y_train, y_valid, C=0.01)
    
    hist.append({
        "train_size"  : train_size,
        "ulm_score_1" : ulm_score_1,
        "ulm_score_2" : ulm_score_2,
        "ulm_score_3" : ulm_score_3,
        "bow_score_1" : bow_score_1,
        "bow_score_2" : bow_score_2,
        "bow_score_3" : bow_score_3,
    })
    print(hist[-1])


hist = pd.DataFrame(hist)

_ = plt.plot(train_sizes, hist.ulm_score_1, marker='o')
_ = plt.plot(train_sizes, hist.ulm_score_2, marker='o')
_ = plt.plot(train_sizes, hist.ulm_score_3, marker='o')
_ = plt.plot(train_sizes, hist.bow_score_1, marker='+')
_ = plt.plot(train_sizes, hist.bow_score_2, marker='+')
_ = plt.plot(train_sizes, hist.bow_score_3, marker='+')
_ = plt.xscale('log')
_ = plt.xlabel('train_size')
_ = plt.ylabel(metric)
_ = plt.title('%s (num_class=%d)' % (args.df_path, len(set(y_train))))
show_plot()
