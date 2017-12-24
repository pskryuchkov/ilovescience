#!/usr/bin/env python

# Frequent analisys with tf-idf. List of topics required (../topics/<section>.txt)
# Usage: 'python freq.py cond-mat.17'


from collections import defaultdict, Counter
from gensim.models import Phrases
from sys import argv, path
import numpy as np
import pickle
import os

path.insert(0, os.path.dirname(
                    os.path.realpath(__file__)) + '/extra')
import shared
import config


volume = None
n_results = 10
check_unrelevant = True
n_proc_articles = config.n_articles


class Volume:
    def __init__(self, section, year, month):
        self.section = section
        self.year = str(year).zfill(2)
        self.month = str(month).zfill(2)


def file_counter(lines, term):
    cnt = 0
    for line in lines:
        cnt += line.count(term)

    return cnt


def group_articles(path, terms_l, min_cnt=5,
                   return_unrelevant=False):

    articles_d = defaultdict(list)

    if not os.path.isfile('cache/{}.cache'.format(volume)):
        files_list = shared.random_glob(path, n_proc_articles)
    else:
        with open('cache/{}.cache'.format(volume), 'rb') as f:
            d = pickle.load(f)
            files_list = d.keys()

    for i, file in enumerate(files_list):
        is_relevant = False
        text = open(file).readlines()

        for term in terms_l:

            if file_counter(text, term) >= min_cnt:

                if not is_relevant: is_relevant = True
                articles_d[term].append(file)

        if return_unrelevant and not is_relevant:
            articles_d["uncovered"].append(file)

    return articles_d


def get_unique_articles(articles, terms, articles_dict):     # FIXME
    all_articles = [x for y in articles_dict.values() for x in y]

    unique_articles = [x for x in Counter(all_articles).keys()
                       if Counter(all_articles)[x] == 1]

    term_unique_articles = [[] for z in terms]

    for article in unique_articles:

        for term in terms:

            if article in articles[term]:
                term_unique_articles[terms.index(term)] += [article]

    return term_unique_articles


def tf_idf(*args):
    global_counter = Counter()

    for counter_d in args:
        global_counter += Counter(counter_d)

    result = []
    for counter_d in args:
        subresult = {}
        num_words = sum(counter_d.values())

        for key in counter_d.keys():
            subresult[key] = (1.0 * counter_d[key] / num_words) * \
                             np.log(1.0 + 1.0 * len(args) / global_counter[key])

        result.append(subresult)
    return result


@shared.save_excel(config.freq_stat)
def calc_stat(labels, topics_texts):
    count_dicts = []
    for topic_content in topics_texts:
        cnt_d = sum(map(Counter, topic_content), Counter())

        # python "feature": counting zero string
        if '' in cnt_d.keys():
            del cnt_d['']

        count_dicts.append(dict(cnt_d))

    top = tf_idf(*count_dicts)
    sat_base = []

    for v in range(len(count_dicts)):
        word_sats = []

        for h, key in enumerate(top[v]):

            if key not in stop_list:
                word_sats.append([key,
                                  round(top[v][key], 6),
                                  count_dicts[v][key]])

        sat_base.append(word_sats)


    return shared.MultiTable(volume + "/topics", sat_base, labels).sort(col_idx=1)


@shared.save_csv(config.freq_stat)
def calc_corr(terms, articles):
    unique_pairs = [[i, j] for j in range(len(terms))
                    for i in range(j)]

    corr_vals = []
    for pair in unique_pairs:
        inter_one = list(set(articles[terms[pair[0]]])
                         & set(articles[terms[pair[1]]]))

        inter_two = list(set(articles[terms[pair[1]]])
                         & set(articles[terms[pair[0]]]))

        corr_vals.append([terms[pair[0]], terms[pair[1]],
                          max(1.0 * len(inter_one) /
                              len(articles[terms[pair[0]]]),

                              1.0 * len(inter_two) /
                              len(articles[terms[pair[1]]]))])
    keys = set([])

    for x in corr_vals:
        keys.add(x[0])
        keys.add(x[1])

    n = len(keys)

    d = dict(zip(list(keys), range(n)))

    corr = np.identity(n) * 1.0

    for rec in corr_vals:
        x = d[rec[0]]
        y = d[rec[1]]
        corr[y][x] = rec[2]
        corr[x][y] = rec[2]

    inv_d = {v: k for k, v in d.iteritems()}

    table = list()
    table.append([""] + inv_d.values())

    for y in range(n):
        row = [round(corr[x, y], 3) for x in range(n)]
        table.append([inv_d[y]] + row)

    return shared.SingleTable(volume + "/terms_similar", table)


@shared.save_csv(config.freq_stat)
def calc_unique(terms, articles, unique_articles):
    table = [["term", "n", "unique"]]

    for p in range(len(terms)):
        table.append([terms[p],
                      len(unique_articles[p]),
                      round(1.0 * np.float32(len(unique_articles[p])) / len(articles[terms[p]]), 2)])

    return shared.SingleTable(volume + "/terms_unique", table)


def main(arxiv):
    terms = shared.get_lines("../topics/{}.txt".format(arxiv.section))

    dest_path = config.freq_stat + "{}/".format(volume)
    shared.check_dir(dest_path)

    if check_unrelevant:
        e_terms = terms + ["uncovered"]
    else:
        e_terms = terms

    print("Getting relevant articles...")
    articles = group_articles("../arxiv/{0}/{1}/"
                                .format(arxiv.section, arxiv.year),
                                    terms, return_unrelevant=check_unrelevant)

    unique_articles = get_unique_articles(articles, e_terms, articles)

    print("Coverage:", round(1.0 * sum([len(articles[x]) for x in articles.keys()
                                        if x != "uncovered"]) / n_proc_articles, 2))

    print("Unique coverage:", round(1.0 * (sum([len(x) for x in unique_articles])
                                           - int(check_unrelevant)
                                           * len(unique_articles[-1])) / n_proc_articles, 2))

    calc_unique(terms, articles, unique_articles)

    calc_corr(terms, articles)

    topics_texts = []

    for g in range(len(unique_articles)):
        topic_sentences = []

        # FIXME
        if not os.path.isfile('cache/{}.cache'.format(volume)):
            for file in unique_articles[g]:
                text = " ".join(shared.line_filter(
                                    shared.ascii_normalize(
                                        open(file, "r").readlines()), min_length=4)).split(" ")

                topic_sentences.append(text)
        else:
            d = {}
            with open('cache/{}.cache'.format(volume), 'rb') as f:
                d = pickle.load(f)

            for file in unique_articles[g]:
                text = d[file][0].split()
                topic_sentences.append(text)

        topics_texts.append(topic_sentences)

    if config.biGram:
        print("Searching for bigrams...")

        bigram_transformer = Phrases([sentence for topic_content in topics_texts
                                                for sentence in topic_content])

        topics_texts = [list(bigram_transformer[topic_content])
                                for topic_content in topics_texts]

    print("Calculating tf-idf...")

    calc_stat(e_terms, topics_texts)


def arg_run():
    if len(argv) < 2:
        print("Error: too few arguments")
    elif len(argv) > 3:
        print("Error: too many arguments")
    else:
        global volume

        if "-d" in argv:
            global n_proc_articles
            n_proc_articles = config.n_articles_debug

        volume = argv[1]

        section, year_s = volume.split(".")
        year = int(year_s)

        arxiv_vol = Volume(section, year, 0)

        shared.create_dir(config.freq_stat)
        main(arxiv_vol)


if __name__ == "__main__":
    stop_list = shared.get_lines("extra/stoplist.txt")
    arg_run()
