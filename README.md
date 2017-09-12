# ilovescience

Set of scripts for arxiv.org articles text mining

### Scripts
`craw.py` loads article

`lda.py` extract keywords

`freq.py` runs frequent analisys

`ref.py` counts references

`wotvec.py` console for semantic similar words searching

### Usage 
#### crawl.py, lda.py, freq.py, ref.py

`<script> <section>.<year>.<month>`

#### Example

`python freq.py cond-mat.16.03`

#### wotvec.py
`-t` topics word vectors info

`-b` build word vectors

`-c` word2vec console