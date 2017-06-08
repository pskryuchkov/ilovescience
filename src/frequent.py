from collections import defaultdict, Counter
from unidecode import unidecode
from numpy import log
from glob import glob
import nltk
import re
import os


month = 04
year = 16

stat_path = "../stat/frequency/"
n_articles = 1500
max_n_print = 30
n_results = 100


def file_counter(lines, term):
    cnt = 0
    for line in lines: cnt += line.count(term)
    return cnt


def base_counter(path, terms_l):
    counter_d = dict.fromkeys(terms, 0)

    for i, file in enumerate(glob(path+"*.txt")):
        text = open(file).readlines()

        for term in terms_l:
            counter_d[term] += file_counter(text, term)

    return counter_d


def get_topics(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def ascii_normalize(text):
    # .decode("ascii", "ignore")
    return [unidecode(line.decode("utf-8")) for line in text]


def get_relevant_articles(path, terms_l, min_cnt=5):
    articles_d = defaultdict(list)

    for i, file in enumerate(glob(path+"*.txt")[:n_articles]):
        text = open(file).readlines()

        for term in terms_l:
            if file_counter(text, term) >= min_cnt:
                articles_d[term].append(file)

    return articles_d, len(glob(path+"*.txt"))


def get_unique_articles(articles_dict):     # FIXME
    all_articles = [x for y in articles_dict.values() for x in y]
    unique_articles = [x for x in Counter(all_articles).keys() if Counter(all_articles)[x] == 1]

    term_unique_articles = [[] for z in terms]
    for article in unique_articles:
        for term in terms:
            if article in articles[term]: term_unique_articles[terms.index(term)] += [article]
    return term_unique_articles


def line_filter(text, min_length=4):  # FIXME
    filtered = []
    exp1 = re.compile('==')
    exp2 = re.compile('{.*}')
    for line in text:
        nline = line.strip().replace(",", "")\
                            .replace(";", "")\
                            .replace(":", "")\
                            .replace('"', "")\
                            .replace('(', "")\
                            .replace(')', "")
        if len(exp1.findall(nline)) == 0 \
                and len(exp2.findall(nline)) == 0:
            nline = " ".join([x for x in nline.split()
                              if len(x) >= min_length])
            filtered.append(nline.lower())
    return filtered


def tf_idf(*args):
    #global_counter = Counter()
    #for j in range(len(args)):
    #    sub_counter = Counter({k : 1 for k in args[j].keys()})
    #    global_counter += sub_counter

    global_counter = Counter()
    for counter_d in args: global_counter += Counter(counter_d)

    result = []
    for counter_d in args:
        subresult = {}
        num_words = sum(counter_d.values())
        for key in counter_d.keys():
            subresult[key] = (1.0 * counter_d[key] / num_words) * log(1.0 + 1.0 * len(args) / global_counter[key])

        result.append(subresult)
    return result


def add_dicts(dict1, dict2):
    sum_dict = {}

    for key in list(set(dict1) | set(dict2)):
        val1 = 0
        val2 = 0
        if key in dict1: val1 = dict1[key]
        if key in dict2: val2 = dict2[key]
        sum_dict[key] = val1 + val2

    return sum_dict


terms = get_topics("../topics.txt")

print "Getting relevant articles..."
articles, n_articles = get_relevant_articles("../arxiv/cond-mat/{0}/{1:02d}/".format(year, month), terms)

unique_articles = get_unique_articles(articles)

print "Coverage:", round(1.0 * sum([len(x) for x in articles.values()]) / n_articles, 2)
print "Unique coverage:", round(1.0 * sum([len(x) for x in unique_articles]) / n_articles, 2)

print "-" * 28
print '%-14s %-4s %s' % ('term', 'n', 'unique')
print "-" * 28

for p in range(len(terms)):
    print  '%-14s %-4i %f' % (terms[p], len(unique_articles[p]), \
        round(1.0 * len(unique_articles[p]) / len(articles[terms[p]]), 2))

print "-" * 28
print "Correlations"
unique_pairs = [[i, j] for j in range(len(terms)) for i in range(j)]

print "-" * 34
for pair in unique_pairs:
    inter_one = list(set(articles[terms[pair[0]]]) & set(articles[terms[pair[1]]]))
    inter_two = list(set(articles[terms[pair[1]]]) & set(articles[terms[pair[0]]]))
    print '%-12s %-12s %f' % (terms[pair[0]], terms[pair[1]], max(1.0 * len(inter_one) / len(articles[terms[pair[0]]]),
                                            1.0 * len(inter_two) / len(articles[terms[pair[1]]])))

print "-" * 34

print "Counting words..."
count_base = []
for g in range(len(unique_articles)):
    cnt_d = Counter()
    for file in unique_articles[g]:
        text = " ".join(line_filter(
                            ascii_normalize(
                                open(file, "r").readlines()))).lower().split(" ")

        cnt_d += Counter(text)

    if '' in cnt_d.keys(): del cnt_d['']  # python "feature": counting zero string
    count_base.append(dict(cnt_d))

print "Calculating tf-idf..."
top = tf_idf(*count_base)

stop_list = set(nltk.corpus.stopwords.words("english"))

if not os.path.isdir(stat_path): os.makedirs(stat_path)

for v in range(len(unique_articles)):

    report = open("{}{}.csv".format(stat_path, terms[v]), "w+")
    report.write("sep=,\n")

    print "-" * 22
    print terms[v].upper()
    print "-" * 22

    for h, elem in enumerate(sorted(top[v],
                                    key=top[v].get, reverse=True)[:n_results]):
        if elem not in stop_list:
            if h < max_n_print:
                print  '%-12s %f %d' % (elem, round(top[v][elem], 6), count_base[v][elem])
            elif h == max_n_print: print "..."

            report.write("{}, {}, {}\n".format(elem, round(top[v][elem], 6),
                         count_base[v][elem]))

    report.close()