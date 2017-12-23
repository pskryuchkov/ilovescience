#!/usr/bin/env python

# Keywords extracting (latent dirichlet allocation used)
# Usage: 'python lada.py cond-mat.17'


from gensim.models import LdaMulticore, Phrases
from gensim import corpora
from sys import argv, path
import logging
import pickle
import random
import os

path.insert(0, os.path.dirname(
                    os.path.realpath(__file__)) + '/extra')
import shared
import config


volume = None
n_proc_articles = config.n_articles


class Volume:
    def __init__(self, section, year, month):
        self.section = section
        self.year = str(year).zfill(2)
        self.month = str(month).zfill(2)


def prepare_sentences(file_list, n_articles):
    base = []
    d = {}

    if not os.path.isfile('cache/{}.cache'.format(volume)):
        for g, file in enumerate(file_list[:n_articles]):
            print("{}/{} {}".format(g + 1, n_articles, shared.fn_pure(file)))
            text = " ".join(shared.line_filter(
                                shared.ascii_normalize(
                                    open(file, "r").readlines()), min_length=4)).lower().split(".")
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


def calculate_keys(vol, n_top, n_pass, cache_corpus=True,
                   cache_model=True):

    texts_path = "../arxiv/{0}/{1}/".format(vol.section, vol.year)

    if not os.path.isdir(texts_path):
        raise Exception('There is no such path: {}'.format(texts_path))

    files_list = shared.random_glob(texts_path, n_proc_articles)

    texts = prepare_sentences(files_list, n_proc_articles)

    print("Searching for bigrams...")

    if config.biGram:
        bigram_transformer = Phrases(texts, min_count=10)
        texts = list(bigram_transformer[texts])

    texts = shared.plural_filter(texts)

    print("Building corpus..")
    dictionary = corpora.Dictionary(texts)

    dictionary.filter_extremes(no_below=20)

    corpus = [dictionary.doc2bow(text) for text in texts]
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    print("Running LDA...")
    lda = LdaMulticore(corpus, num_topics=n_top, id2word=dictionary,
                       workers=4, passes=n_pass, iterations=400, eval_every=None)

    if cache_corpus:
        with open(config.lda_stat + "{0}.corpus".format(volume), 'wb') as f:
            pickle.dump(corpus, f)

        with open(config.lda_stat + "{0}.dict".format(volume), 'wb') as f:
            pickle.dump(texts, f)

    if cache_model:
        lda.save("{0}{1}".format(config.lda_stat, volume))
    return lda


def topics(arxiv, n_top=30, n_pass=30, short_keylist=True, choice_mode="f"):
    lda = calculate_keys(arxiv, n_top, n_pass)

    table = []
    for record in lda.show_topics(num_topics=n_top, num_words=12):
        data = record[1].split(" + ")

        topic = []
        for s_record in data:
            weight, word = s_record.split("*")
            topic.append([weight, word[1:-1]])

        table.append(topic)

    if short_keylist:
        with open("../topics/{}.txt".format(arxiv.section), "w") as f:
            for topic in table:
                if choice_mode == "r":
                    choice = random.choice(topic)
                    f.write("{}\n".format(choice[1]))
                elif choice_mode == "f":
                    f.write("{}\n".format(topic[0][1]))
    else:
        report = open(config.lda_stat + "{0}.keys.csv".format(volume), "w+")
        report.write("sep=,\n")

        for c, topic in enumerate(table):
            report.write("topic #{}\n".format(c + 1))

            for word in topic:
                report.write("{0},{1}\n".format(word[0], word[1]))

        report.close()


def arg_run():
    if len(argv) < 2:
        print("Error: too few arguments")
    elif len(argv) > 4:
        print("Error: too many arguments")
    else:
        if "-d" in argv:
            global n_proc_articles
            n_proc_articles = config.n_articles_debug

        global volume
        volume = argv[1]

        section, year_s = volume.split(".")
        year = int(year_s)
        arxiv_vol = Volume(section, year, 0)

        n_topics = 15
        n_passes = 30

        short_flag = False
        if "-s" in argv:
            short_flag = True
            n_topics = 10
            n_passes = 15

            shared.create_dir(config.lda_stat)

        # optimal values: n_topics and n_passes ~ 30
        topics(arxiv_vol, n_topics, n_passes, short_keylist=short_flag)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    stoplist = [x.rstrip() for x in open("extra/stoplist.txt", "r").readlines()]
    arg_run()
