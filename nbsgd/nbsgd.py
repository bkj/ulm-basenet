#!/usr/bin/env python

"""
    nbsgd.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils import data

from basenet import BaseNet
from basenet.hp_schedule import HPSchedule
from basenet.helpers import to_numpy, set_seeds

# --
# Helpers

def calc_r(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))

# --
# IO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-train', type=str, default='./data/aclImdb/X_train.npy')
    parser.add_argument('--x-train-words', type=str, default='./data/aclImdb/X_train_words.npy')
    parser.add_argument('--y-train', type=str, default='./data/aclImdb/y_train.npy')
    
    parser.add_argument('--x-test', type=str, default='./data/aclImdb/X_test.npy')
    parser.add_argument('--x-test-words', type=str, default='./data/aclImdb/X_test_words.npy')
    parser.add_argument('--y-test', type=str, default='./data/aclImdb/y_test.npy')
    
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--lr-max', type=float, default=0.02)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--batch-size', type=int, default=256)
    
    parser.add_argument('--vocab-size', type=int, default=200000)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)

# --
# IO

print('nbsgd.py: making dataloaders...', file=sys.stderr)

X_train         = np.load(args.x_train).item()
X_train_words   = np.load(args.x_train_words).item()
y_train         = (np.load(args.y_train) == 'pos').astype(int)
cl_train_df     = pd.read_csv('../runs/0/classifier/preds/classifier-train', sep='\t', header=None)
cl_train_logits = cl_train_df.values

X_lm       = np.load('./data/aclImdb/X_lm.npy').item()
X_lm_words = np.load('./data/aclImdb/X_lm_words.npy').item()
y_lm       = np.load('./data/aclImdb/y_lm.npy')
lm_df      = pd.read_csv('../runs/0/classifier/preds/lm-train', sep='\t', header=None)
lm_logits  = lm_df.values

train_dataset = torch.utils.data.dataset.TensorDataset(
    torch.cat([
        torch.from_numpy(X_train_words.toarray()).long(),
        torch.from_numpy(X_lm_words.toarray()).long(),
    ], dim=0),
    torch.cat([
        torch.cat([
            torch.from_numpy(y_train).float().view(-1, 1),
            torch.from_numpy(cl_train_logits).float(),
        ], dim=-1),
        torch.cat([
            torch.from_numpy(y_lm).float().view(-1, 1),
            torch.from_numpy(lm_logits).float(),
        ], dim=-1)
    ], dim=0)
)

X_test        = np.load(args.x_test).item()
X_test_words  = np.load(args.x_test_words).item()
y_test        = (np.load(args.y_test) == 'pos').astype(int)



test_dataset = torch.utils.data.dataset.TensorDataset(
    torch.from_numpy(X_test_words.toarray()).long(),
    torch.from_numpy(y_test).long(),
)

dataloaders = {
    "train" : torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
    "test"  : torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
}

n_classes = int(y_train.max()) + 1
r = np.column_stack([calc_r(i, X_train, y_train) for i in range(n_classes)])

# --
# Model definition

def loss_fn(x, ys, alpha=1.0, beta=0.5, T=1.5):
    if len(ys.shape) > 1:
        y = ys[:,0].long()
    else:
        y = ys.long()
    
    if len(ys.shape) > 1:
        ce_loss = F.cross_entropy(x[(y >= 0).nonzero().squeeze()], y[y >= 0])
        
        log_X          = F.log_softmax(x, dim=-1)
        y_soft_softmax = F.softmax(ys[:,1:] / T, dim=-1)
        soft_loss      = - (y_soft_softmax * log_X).sum(dim=-1).mean()
        return beta * ce_loss + alpha * soft_loss
    else:
        return F.cross_entropy(x, y)

class DotProdNB(BaseNet):
    def __init__(self, vocab_size, n_classes, r, w_adj=0.4, r_adj=10):
        
        super().__init__(loss_fn=loss_fn)
        
        # Init w
        self.w = nn.Embedding(vocab_size + 1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1, 0.1)
        self.w.weight.data[0] = 0
        
        # Init r
        self.r = nn.Embedding(vocab_size + 1, n_classes)
        self.r.weight.data = torch.Tensor(np.concatenate([np.zeros((1, n_classes)), r])).cuda()#.to(torch.device('cuda'))
        self.r.weight.requires_grad = False
        
        self.w_adj = w_adj
        self.r_adj = r_adj
        
    def forward(self, feat_idx):
        w = self.w(feat_idx) + self.w_adj
        r = self.r(feat_idx)
        
        x = (w * r).sum(dim=1)
        x =  x / self.r_adj
        return x

# --
# Define model

print('nbsgd.py: initializing model...', file=sys.stderr)

model = DotProdNB(args.vocab_size, n_classes, r).to('cuda')
model.verbose = False
print(model, file=sys.stderr)

# --
# Initializing optimizer 

lr_scheduler = getattr(HPSchedule, args.lr_schedule)(hp_max=args.lr_max, epochs=args.epochs)
model.init_optimizer(
    opt=torch.optim.Adam,
    params=[p for p in model.parameters() if p.requires_grad],
    hp_scheduler={"lr" : lr_scheduler},
    weight_decay=args.weight_decay,
)

# --
# Train

print('nbsgd.py: training...', file=sys.stderr)
t = time()
for epoch in range(args.epochs):
    train = model.train_epoch(dataloaders, mode='train', compute_acc=False)
    test  = model.eval_epoch(dataloaders, mode='test')
    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.hp['lr'],
        "test_acc"  : float(test['acc']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

model.verbose = True
t = time()
_ = model.predict(dataloaders, mode='train')
print(time() - t)

model.save('weights')
