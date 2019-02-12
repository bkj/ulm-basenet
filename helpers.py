#!/usr/bin/env python

"""
    helpers.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from basenet.helpers import to_numpy

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


def extract_features(model, dataloaders, mode):
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