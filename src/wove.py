#!/usr/bin/env python

# Word2vec algorithm applying to scientific texts

# Calculating word vectors
# python wove.py cond-mat.17 -b

# Topics word vectors info
# python wove.py cond-mat.17 -t

# Word2vec console (pre-calclulated wordvectors required)
# python wove.py cond-mat.17


from gensim.models import Word2Vec, Phrases
from sys import argv, path
import logging
import pickle
import re
import os

path.insert(0, os.path.dirname(
                    os.path.realpath(__file__)) + '/extra')
import shared
import config


min_count = 10
size = 300
window = 10

n_ass = 10

n_proc_articles = config.n_articles

# FIXME
texts_path = "../arxiv/{0}/{1}/{2:02d}/*.txt"
vec_path = config.vec_stat + "{0}"
topics_path = ""

volume = None


# do we need it?
def break_remove(raw_text):
    text = " ".join(raw_text)
    break_expr = re.compile(r'(?![A-Za-z0-9]+)\s*-\s*(?=[A-Za-z0-9]+)')
    text = break_expr.sub("", text)
    return text.split("\n")


def prepare_sentences(file_list, n_articles):
    base = []
    d = {}

    if not os.path.isfile('cache/{}.cache'.format(volume)):
        for g, file in enumerate(file_list[:n_articles]):

            print("{}/{} {}".format(g + 1, n_articles, shared.fn_pure(file)))

            text = " ".join(shared.line_filter(
                                shared.ascii_normalize(
                                    open(file, "r").readlines()), min_length=3)).lower().split(".")

            d[file] = text
            base += [x.split() for x in text]

        with open('cache/{}.cache'.format(volume), 'wb') as f:
            pickle.dump(d, f)

    else:

        with open('cache/{}.cache'.format(volume), 'rb') as f:
            d = pickle.load(f)

        for g, file in enumerate(d.keys()):
            print("{}/{} {}".format(g + 1, n_articles, shared.fn_pure(file)))
            text = d[file]

            base += [x.split() for x in text]
    return base


def build_word_vec(show_log=True):

    section, year = volume.split(".")
    texts_path = "../arxiv/{0}/{1}/".format(section, year)

    files_list = shared.random_glob(texts_path, n_proc_articles)
    sentences = prepare_sentences(files_list, n_proc_articles)

    if show_log:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    if config.biGram:
        bigram_transformer = Phrases(sentences, min_count=10)
        sentences = list(bigram_transformer[sentences])

    sentences = shared.plural_filter(sentences)

    return Word2Vec(sentences, min_count=min_count, size=size, window=window, workers=4)


def replics(model, target_word, topn=10):
    return {word: model.vocab[word].count
                           for word in model.vocab.keys()
                           if word.find(target_word) == 0}


def topics_interset(model):
    topics = shared.get_lines(topics_path)

    topics_normalized = []
    for topic in topics:
        replics_counter = replics(model, topic)
        topics_normalized.append(max(replics_counter,
                                     key=replics_counter.get))

    pairs = [[i, j] for j in range(len(topics_normalized)) for i in range(j)]

    for pair in pairs:
        similar = model.most_similar(positive=[topics_normalized[pair[0]],
                                               topics_normalized[pair[1]]], topn=n_ass)
        cnt = sum([1 for record in similar if record[1] >= 0.6])
        print("{} {} ({})".format(topics_normalized[pair[0]], topics_normalized[pair[1]], cnt))
        for x in sorted(similar, key=(lambda x: x[1]), reverse=True):
            if x[1] >= 0.6:
                print("  {} {}".format(x[0], round(x[1],2)))


def topics_normalize(model, raw_topics):
    norm_topics = []
    for word in raw_topics:
        r = replics(model, word)
        if r is not None:
            print("{} -> {} ({})".format(word, max(r, key=r.get), r[max(r, key=r.get)]))
            norm_topics.append(max(r, key=r.get))
        else:
            norm_topics.append(None)
            print(word, "None")

    return norm_topics


def topics_info(word_vec_path):
    model = Word2Vec.load(word_vec_path)

    topics = topics_normalize(model, shared.get_lines(topics_path))

    for word in topics:
        if word in model.vocab.keys():
            print(word)
            for neighbor in model.most_similar(positive=[word], topn=n_ass):

                print("  '{}', {}, {}".format(neighbor[0], round(neighbor[1], 2),
                                              model.vocab[neighbor[0]].count))
        else:
            print(word, "None")


def vocab(word_vec_path):
    model = Word2Vec.load(word_vec_path)
    keys = model.vocab.keys()
    keys.sort()
    return keys


def console(path):
    model = Word2Vec.load(path)

    print("Vocabulary length: {} words".format(len(model.vocab.keys())))
    print("Minimal count: {} words".format(model.min_count))

    while True:
        query = raw_input('> ')
        if query in model.vocab.keys():
            for record in model.most_similar(positive=[query], topn=10):
                print("{}, {}, {}".format(record[0], round(record[1], 2), model.vocab[record[0]].count))
        else:
            print("There is no such word: '{}'".format(query))


def arg_run():
    if len(argv) < 2:
        raise Exception("Too few arguments")

    if len(argv) > 4:
        raise Exception("Too many arguments")

    if "-d" in argv:
        global n_proc_articles
        n_proc_articles = config.n_articles_debug

    b_flag = "-b" in argv
    t_flag = "-t" in argv

    global volume
    volume = argv[1]

    s, y = volume.split(".")
    y = int(y)

    global topics_path
    topics_path = "../topics/{}.txt".format(s)

    if not b_flag and not t_flag:
        # console
        print("Word2vec on arxiv {}.{:02d}".format(y))
        console(vec_path.format(volume))
    else:
        if b_flag:
            # build
            model = build_word_vec()
            model.save(vec_path.format(volume))
        elif t_flag:
            # topic info
            topics_info(vec_path.format(volume))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    shared.create_dir(config.vec_stat)
    arg_run()
