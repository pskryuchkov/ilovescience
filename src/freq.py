# Frequent analisys (with tf-idf). List of topics required (extra/topics.txt)
# Usage: 'python freq.py cond-mat.16.03'
# FIXME usage: './freq.py cond-mat.17'

# Tested for Anaconda Python 2.7.13
# If script doesn't work, check your Python interpteter version.

from collections import defaultdict, Counter
from gensim.models import Phrases
from unidecode import unidecode
from sys import argv
from os import chdir
import numpy as np
import fnmatch
import pickle
import random
import errno
import re
import os


volume = None
biGram = False
n_results = 10
n_articles = 1000
n_articles_debug = 100

check_unrelevant = True
stat_path = "../stat/frequency/"


class Volume:
    def __init__(self, section, year, month):
        self.section = section
        self.year = str(year).zfill(2)
        self.month = str(month).zfill(2)


class SingleTable:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def sort(self, col_idx, reverse=True):
        sorted_table = sorted(self.content,
                              key=lambda x: x[col_idx],
                              reverse=reverse)

        return SingleTable(self.name, sorted_table)

    def to_multi(self):
        return MultiTable(self.name, [self.content], [self.name])


class MultiTable:
    def __init__(self, name, content,
                 labels):

        self.name = name
        self.labels = labels
        self.content = content

    def sort(self, col_idx, reverse=True):
        sorted_table = []

        for sheet in self.content:
            sorted_table.append(sorted(sheet,
                                       key=lambda x: x[col_idx],
                                       reverse=reverse))

        return MultiTable(self.name, sorted_table, self.labels)


def create_dir(dn):
    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_csv(path="", sep=","):
    def decorator(func):
        def inner(*args):

            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            for label, sheet in zip(table.labels, table.content):

                f = open(path + label + ".csv", "w")
                f.write("sep={}\n".format(sep))

                for line in sheet:
                    f.write(sep.join(map(lambda x: str(x), line)) + "\n")

                f.close()

            return table
        return inner
    return decorator


def save_excel(path=""):
    def decorator(func):
        def inner(*args):

            from openpyxl import Workbook
            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            wb = Workbook()

            first_sheet = True
            for label, sheet in zip(table.labels, table.content):

                if first_sheet:
                    first_sheet = False
                    ws = wb.active
                    ws.title = label
                else:
                    ws = wb.create_sheet(label)

                for line in sheet:
                    ws.append(line)

            wb.save(path + table.name + ".xlsx")
            wb.close()

            return table
        return inner
    return decorator


def console_table(n_print=6, n_sep=30):
    def decorator(func):
        def inner(*args):

            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            for label, sheet in zip(table.labels, table.content):

                print "-" * n_sep
                print label.upper()
                print "-" * n_sep

                for j, line in enumerate(sheet):
                    if j < n_print:
                        print ", ".join(map(lambda x: str(x), line))
                    else:
                        print "..."
                        break

            return table
        return inner
    return decorator


def file_counter(lines, term):
    cnt = 0
    for line in lines:
        cnt += line.count(term)

    return cnt


def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def ascii_normalize(text):
    return [unidecode(line.decode("utf-8")) for line in text]


def group_articles(path, terms_l, min_cnt=5,
                   return_unrelevant=False):

    articles_d = defaultdict(list)

    if not os.path.isfile('extra/{}.cache'.format(volume)):
        files_list = random_glob(path, n_articles)
    else:
        with open('extra/{}.cache'.format(volume), 'rb') as f:
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


def line_filter(text, min_length=4):
    brackets = re.compile(r'{.*}') # remove formulas
    alphanum = re.compile(r'[\W_]+')

    filtered = []
    for line in text:
        nline = brackets.sub(' ', line).strip()
        nline = alphanum.sub(' ', nline).strip()

        nline = " ".join([x for x in nline.split()
                                if len(x) >= min_length # FIXME: empty strings
                                and not x.isdigit()
                                and not x in stop_list])

        filtered.append(nline.lower())

    return filtered


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


@save_excel(stat_path)
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
    #from pprint import pprint
    #pprint(sat_base)
    return MultiTable(volume + "/topics", sat_base, labels).sort(col_idx=1)


@save_csv(stat_path)
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

    return SingleTable(volume + "/terms_similar", table)


@save_csv(stat_path)
def calc_unique(terms, articles, unique_articles):
    table = [["term", "n", "unique"]]

    for p in range(len(terms)):
        table.append([terms[p],
                      len(unique_articles[p]),
                      round(1.0 * len(unique_articles[p]) / len(articles[terms[p]]), 2)])

    return SingleTable(volume + "/terms_unique", table)


# FIXME: remove this
def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def random_glob(path, n_files, mask="*.txt"):
    file_list = []

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            file_list.append(os.path.join(root, filename))

    random.shuffle(file_list)
    return file_list[:n_files]


def main(arxiv):
    terms = get_lines("../topics/{}.txt".format(arxiv.section))

    dest_path = stat_path + "{}/".format(volume)
    check_dir(dest_path)

    if check_unrelevant:
        e_terms = terms + ["uncovered"]
    else:
        e_terms = terms

    print "Getting relevant articles..."
    articles = group_articles("../arxiv/{0}/{1}/"
                                .format(arxiv.section, arxiv.year),
                                    terms, return_unrelevant=check_unrelevant)

    unique_articles = get_unique_articles(articles, e_terms, articles)

    print "Coverage:", round(1.0 * sum([len(articles[x]) for x in articles.keys()
                                        if x != "uncovered"]) / n_articles, 2)

    print "Unique coverage:", round(1.0 * (sum([len(x) for x in unique_articles])
                                           - int(check_unrelevant)
                                           * len(unique_articles[-1])) / n_articles, 2)

    calc_unique(terms, articles, unique_articles)

    calc_corr(terms, articles)

    topics_texts = []

    for g in range(len(unique_articles)):
        topic_sentences = []

        # FIXME
        if not os.path.isfile('extra/{}.cache'.format(volume)):
            for file in unique_articles[g]:
                text = " ".join(line_filter(
                                    ascii_normalize(
                                        open(file, "r").readlines()))).split(" ")

                topic_sentences.append(text)
        else:
            d = {}
            with open('extra/{}.cache'.format(volume), 'rb') as f:
                d = pickle.load(f)

            for file in unique_articles[g]:
                text = d[file][0].split()
                topic_sentences.append(text)

        topics_texts.append(topic_sentences)

    if biGram:
        print("Searching for bigrams...")
        
        # FIXME: bigram transformer returns empty list 
        bigram_transformer = Phrases([sentence for topic_content in topics_texts
                                                for sentence in topic_content])

        topics_texts = [bigram_transformer[topic_content]
                        for topic_content in topics_texts]

    print "Calculating tf-idf..."

    calc_stat(e_terms, topics_texts)


def arg_run():
    if len(argv) < 2:
        print "Error: too few arguments"
    elif len(argv) > 3:
        print "Error: too many arguments"
    else:
        global volume

        if "-d" in argv:
            global n_articles
            n_articles = n_articles_debug

        volume = argv[1]
        #s, y, m = volume.split(".")
        #y, m = int(y), int(m)

        section, year_s = volume.split(".")
        year = int(year_s)

        arxiv_vol = Volume(section, year, 0)

        create_dir(stat_path)
        #main(s, y, m)
        main(arxiv_vol)


if __name__ == "__main__":
    chdir(os.path.dirname(os.path.realpath(__file__)))
    stop_list = get_lines("extra/stoplist.txt")
    arg_run()

