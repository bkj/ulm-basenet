#!/usr/bin/env python

"""
    ulmfit.py
    
    ULMFIT functions
    
    !! Most of these functions are copied exactly/approximately from the `fastai` library.
        https://github.com/fastai/fastai
    
    !! This will not work on torch>=0.4, due to torch bugs
"""

import re
import sys
import json
import warnings
import numpy as np
from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from basenet import BaseNet, HPSchedule
from basenet.helpers import parameters_from_children

# --
# Helpers

def detach(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return tuple([detach(xx) for xx in x])
    # elif IS_TORCH_04:
    #     return x.detach()
    else:
        return Variable(x.data)


# --
# LM dataloader

class LanguageModelLoader():
    # From `fastai`
    def __init__(self, data, bs, bptt, backwards=False):
        self.bs        = bs
        self.bptt      = bptt
        self.backwards = backwards
        self.data      = self.batchify(data, bs)
        self.i         = 0
        self.iter      = 0
        self.n         = len(self.data)
        
    def batchify(self, data, bs):
        trunc = data.shape[0] - data.shape[0] % bs
        data = np.array(data[:trunc])
        
        data = data.reshape(bs, -1).T
        
        if self.backwards:
            data = data[::-1]
        
        return torch.LongTensor(np.ascontiguousarray(data))
    
    def __iter__(self):
        self.i    = 0
        self.iter = 0
        while (self.i < self.n - 1) and (self.iter < len(self)):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res
        
    def get_batch(self, i, seq_len):
        seq_len = min(seq_len, self.data.shape[0] - 1 - i)
        return self.data[i:(i+seq_len)], self.data[(i+1):(i+seq_len+1)].view(-1)
        
    def __len__(self):
        return self.n // self.bptt - 1

# --
# RNN Encoder

def dropout_mask(x, sz, dropout):
    # From `fastai`
    return x.new(*sz).bernoulli_(1 - dropout)/ (1 - dropout)


class LockedDropout(nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or not self.p:
            return x
        else:
            mask = dropout_mask(x.data, (1, x.shape[1], x.shape[2]), self.p)
            mask = Variable(mask, requires_grad=False)
            return mask * x
    
    def __repr__(self):
        return 'LockedDropout(p=%f)' % self.p


class WeightDrop(torch.nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        super().__init__()
        self.module  = module
        self.weights = weights
        self.dropout = dropout
        
        if isinstance(self.module, torch.nn.RNNBase):
            def noop(*args, **kwargs): return
            self.module.flatten_parameters = noop
        
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))
    
    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)
            
    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)
    
    def __repr__(self):
        return 'WeightDrop(%s)' % self.module.__repr__()


class EmbeddingDropout(nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, embed):
        super().__init__()
        self.embed = embed
        
    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            mask = dropout_mask(self.embed.weight.data, (self.embed.weight.size(0), 1), dropout)
            mask = Variable(mask)
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight
        
        if scale:
            masked_embed_weight = scale * masked_embed_weight
        
        padding_idx = self.embed.padding_idx
        if padding_idx is None:
            padding_idx = -1
        
        # if IS_TORCH_04:
        #     X = F.embedding(words,
        #         masked_embed_weight, padding_idx, self.embed.max_norm,
        #         self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)
        # else:
        return self.embed._backend.Embedding.apply(words,
            masked_embed_weight, padding_idx, self.embed.max_norm,
            self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)
    
    def __repr__(self):
        return 'EmbeddingDropout(%s)' % self.embed.__repr__()


class RNN_Encoder(nn.Module):
    # From `fastai`
    def __init__(self, n_tok, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, initrange=0.1):
        
        super().__init__()
        
        self.emb_sz     = emb_sz
        self.nhid       = nhid
        self.nlayers    = nlayers
        self.dropoute   = dropoute
        self.ndir       = 2 if bidir else 1
        self.batch_size = 1
        
        self.encoder = nn.Embedding(n_tok, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.dropouti = LockedDropout(dropouti)
        
        self.rnns = [
            nn.LSTM(
                input_size=emb_sz if l == 0 else nhid, 
                hidden_size=(nhid if l != nlayers - 1 else emb_sz) // self.ndir,
                num_layers=1, 
                bidirectional=bidir, 
                dropout=dropouth
            ) for l in range(nlayers)
        ]
        self.rnns = [WeightDrop(rnn, dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])
        
        self.encoder.weight.data.uniform_(-initrange, initrange)
    
    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz) // self.ndir
        return Variable(self.weights.new(self.ndir, self.batch_size, nh).zero_(), volatile=not self.training)
        
    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]
    
    def forward(self, x):
        batch_size = x.shape[1]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.reset()
        
        emb = self.encoder_with_dropout(x, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)
        
        raw_output = emb
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])
            
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            
            if l != self.nlayers - 1:
                raw_output = drop(raw_output)
            
            outputs.append(raw_output)
        
        self.hidden = detach(new_hidden)
        return raw_outputs, outputs


# --
# LM classes

class LinearDecoder(nn.Module):
    # From `fastai`
    def __init__(self, in_features, out_features, dropout, decoder_weights=None, initrange=0.1):
        super().__init__()
        
        self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if decoder_weights:
            self.decoder.weight = decoder_weights.weight
        
        self.dropout = LockedDropout(dropout)
    
    def forward(self, input):
        _, x = input
        
        x = self.dropout(x[-1])
        x = x.view(x.size(0) * x.size(1), x.size(2))
        x = self.decoder(x)
        x = x.view(-1, x.size(1))
        
        return x


class LanguageModel(BaseNet):
    def __init__(self, n_tok, emb_sz, nhid, nlayers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True):
        
        def _lm_loss_fn(output, target):
            return F.cross_entropy(output, target)
        
        super().__init__(loss_fn=_lm_loss_fn)
        
        self.encoder = RNN_Encoder(
            n_tok=n_tok,
            emb_sz=emb_sz,
            nhid=nhid,
            nlayers=nlayers,
            pad_token=pad_token,
            dropouth=dropouth,
            dropouti=dropouti,
            dropoute=dropoute,
            wdrop=wdrop,
        )
        
        self.decoder = LinearDecoder(
            in_features=emb_sz,
            out_features=n_tok,
            dropout=dropout,
            decoder_weights=self.encoder.encoder if tie_weights else None
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_layer_groups(self):
        return [
            *zip(self.encoder.rnns, self.encoder.dropouths),
            (self.decoder, self.encoder.dropouti)
        ]
    
    def reset(self):
        _ = [c.reset() for c in self.children() if hasattr(c, 'reset')]
        
    def load_weights(self, wgts):
        tmp = {}
        for k,v in wgts.items():
            k = re.sub(r'^0.', 'encoder.', k)
            k = re.sub(r'^1.', 'decoder.', k)
            tmp[k] = v
        
        self.load_state_dict(tmp)

# --
# Classifier classes


class MultiBatchRNN(RNN_Encoder):
    # From `fastai`
    def __init__(self, bptt, max_seq, predict_only=False, *args, **kwargs):
        self.max_seq      = max_seq
        self.bptt         = bptt
        self.predict_only = predict_only
        super().__init__(*args, **kwargs)
        
    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]
        
    def forward(self, x):
        sl  = x.shape[0]
        _ = [[hh.data.zero_() for hh in h] for h in self.hidden]
        
        raw_outputs, outputs = [], []
        for i in range(0, sl, self.bptt):
            raw_output, output = super().forward(x[i: min(i + self.bptt, sl)])
            if i > (sl - self.max_seq):
                raw_outputs.append(raw_output)
                outputs.append(output)
        
        return self.concat(raw_outputs), self.concat(outputs)


class PoolingLinearClassifier(nn.Module):
    # Adapted from `fastai`
    def __init__(self, layers, drops, predict_only=False):
        super().__init__()
        
        self.predict_only = predict_only
        
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers += [
                nn.BatchNorm1d(num_features=layers[i]),
                nn.Dropout(p=drops[i]),
                nn.Linear(in_features=layers[i], out_features=layers[i + 1]),
                nn.ReLU(),
            ]
        
        self.layers.pop() # Remove last relu
        
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        raw_outputs, outputs = x
        last_raw_output, last_output = raw_outputs[-1], outputs[-1]
        
        x = torch.cat([
            last_output[-1],
            last_output.max(dim=0)[0],
            last_output.mean(dim=0)
        ], 1)
        
        if self.predict_only:
            return self.layers(x)
        else:
            return self.layers(x), last_raw_output, last_output


class TextClassifier(BaseNet):
    # From `fastai`
    def __init__(self, bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, head_layers, head_drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, loss_fn=None, predict_only=False):
        
        super().__init__(loss_fn=loss_fn)
        self.encoder = MultiBatchRNN(
            bptt=bptt,
            max_seq=max_seq,
            predict_only=predict_only,
            n_tok=n_tok,
            emb_sz=emb_sz,
            nhid=n_hid,
            nlayers=n_layers,
            pad_token=pad_token,
            bidir=bidir,
            dropouth=dropouth,
            dropouti=dropouti,
            dropoute=dropoute, 
            wdrop=wdrop,
        )
        
        self.decoder = PoolingLinearClassifier(head_layers, head_drops, predict_only=predict_only)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def get_layer_groups(self):
        return [
            (self.encoder.encoder, self.encoder.dropouti), 
            *zip(self.encoder.rnns, self.encoder.dropouths),
            (self.decoder)
        ]
        
    def reset(self):
        _ = [c.reset() for c in self.children() if hasattr(c, 'reset')]


# --
# Basenet helper

def basenet_train(model, dataloaders, num_epochs, lr_breaks, lr_vals, adam_betas, weight_decay=0, clip_grad_norm=0,  save_prefix=None):
    
    params = [{
        "params" : parameters_from_children(lg, only_requires_grad=True),
    } for lg in model.get_layer_groups()]
    
    model.init_optimizer(
        opt=torch.optim.Adam,
        params=params,
        hp_scheduler={
            "lr" : HPSchedule.piecewise_linear(breaks=lr_breaks, vals=lr_vals)
        },
        betas=adam_betas,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm, # !! Does this work the same way as in fastai?
    )
    
    fitist = []
    t = time()
    for epoch in range(num_epochs):
        train = model.train_epoch(dataloaders, mode='train')
        valid = model.eval_epoch(dataloaders, mode='valid', metric_fns=['n_correct'])
        fitist.append({
            "epoch"      : int(epoch),
            "train_loss" : float(train['loss'][-1]),
            "valid_acc"  : float(valid['acc']),
            "valid_loss" : float(valid['loss'][-1]),
            "elapsed"    : float(time() - t),
        })
        print(json.dumps(fitist[-1]))
        sys.stdout.flush()
        
        if save_prefix is not None:
            torch.save(model.state_dict(), '%s-epoch%d.h5' % (save_prefix, epoch))
    
    return fitist