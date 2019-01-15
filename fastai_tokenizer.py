#!/usr/bin/env python

"""
    fastai_tokenizer.py
    
    Tokenizer from `fastai.text`
    
    !! Have to use this to be compatible w/ the `fastai` pretrained language models
"""

import re
import spacy
import numpy as np
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor


class Tokenizer():
    re_rep      = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')
    re_br       = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    
    def __init__(self, lang='en'):
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])
            
    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.re_br.sub("\n", x))]

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP = ' t_up '
        res = [[TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()] for s in re.findall(r'\w+|\W+', ss)]
        return ''.join(sum(res, []))

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(docs, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(doc) for doc in docs]
    
    @staticmethod
    def proc_all_mp(docs, n_jobs=32, lang='en'):
        chunked_docs = np.array_split(docs, n_jobs)
        with ProcessPoolExecutor(n_jobs) as e:
            return sum(e.map(Tokenizer.proc_all, chunked_docs, [lang] * len(chunked_docs)), [])
