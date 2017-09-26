# ilovescience

Set of scripts for [arxiv.org](https://arxiv.org/) articles text mining

### Scripts

`load.py` loads articles. It may take several hours.

`lada.py` extract keywords ([latent dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) used)

`freq.py` runs frequent analisys (with [tf-idf](https://ru.wikipedia.org/wiki/TF-IDF)). List of topics required (topics.txt).

`cite.py` counts references and show most citied articles.

`wove.py` console for semantic similar words searching ([word2vec](https://ru.wikipedia.org/wiki/Word2vec)). Pre-calclulated wordvectors required.

Articles stored in `.txt` format in `arxiv/<section>/<year>/<month>/` Results stored in `stat` path.

### Usage

`<script> <section>.<year>.<month>`

For word vectors calculating in `wove.py` use -b flag.

### Example

`python lada.py cond-mat.16.03`
