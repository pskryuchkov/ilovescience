# ilovescience

Set of scripts for [arxiv.org](https://arxiv.org/) articles text mining

### Scripts

`articles_crawl.py` loads articles. It may take several hours

`annotations_crawl.py` loads annotations

`lda.py` extracts topics with [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

`terms_cn.py` counts keywords in articles base

`cites.py` counts references and show most citied articles

`word_vec.py` builds [word2vec](https://ru.wikipedia.org/wiki/Word2vec) model

Scripts stored in `src` path. Articles stored in `.txt` format in `arxiv/<section>/<year>/<month>/` Results stored in `stat` path. 

### Usage

`discover.py <section>.<year>` run all analysing scripts

`notes.py <section>.<year>` generate and open Jupyter notebook with calculated statistics
