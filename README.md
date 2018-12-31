### ulm-basenet

Implementation of [ulmfit](https://github.com/fastai/fastai/tree/master/courses/dl2/imdb_scripts) using [basenet](https://github.com/bkj/basenet) wrappers.

Code in `ulmfit.py` is adapted directly from [fastai](https://github.com/fastai/fastai) code.

#### Installation

```
conda create -n ulm_env python=3.6 pip -y
source activate ulm_env

# pytorch
conda install pytorch pytorch=0.3.1 cuda90 -c pytorch -y

# spacy (for tokenization)
conda install -c conda-forge spacy -y
python -m spacy download en

# additional requirements
pip install -r requirements.txt
```

#### Usage

See `./run.sh` for usage.

#### Todo

- Update to pytorch==0.4
  - !! Looked into this on 2018-12-02 -- pytorch==0.4 has a bug that doesn't allow freezing RNNs in the way we need to.  May need to wait for pytorch==1.0 for a fix, so I'm waiting for now.
