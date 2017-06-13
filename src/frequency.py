# texts frequency analysis
# for each term statistics save to '../stat/frequency/<term>.csv'
#
# usage
# python frequency.py cond-mat.16.03

from collections import defaultdict, Counter
from gensim.models import Phrases
from unidecode import unidecode
from numpy import log
from glob import glob
from sys import argv
import re
import os

check_unrelevant = True
biGram = True

stat_path = "../stat/frequency/"
n_articles = 1500
max_n_print = 30
n_results = 100


def file_counter(lines, term):
    cnt = 0
    for line in lines: cnt += line.count(term)
    return cnt


def base_counter(terms, path):
    counter_d = dict.fromkeys(terms, 0)

    for i, file in enumerate(glob(path+"*.txt")):
        text = open(file).readlines()

        for term in terms:
            counter_d[term] += file_counter(text, term)

    return counter_d


def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def ascii_normalize(text):
    # .decode("ascii", "ignore")
    return [unidecode(line.decode("utf-8")) for line in text]


def group_articles(path, terms_l, min_cnt=5, return_unrelevant=False):
    articles_d = defaultdict(list)

    for i, file in enumerate(glob(path+"*.txt")[:n_articles]):
        is_relevant = False
        text = open(file).readlines()

        for term in terms_l:
            if file_counter(text, term) >= min_cnt:
                if not is_relevant: is_relevant = True
                articles_d[term].append(file)

        if return_unrelevant and not is_relevant: articles_d["uncovered"].append(file)

    return articles_d, len(glob(path+"*.txt"))


def get_unique_articles(articles, terms, articles_dict):     # FIXME
    all_articles = [x for y in articles_dict.values() for x in y]
    unique_articles = [x for x in Counter(all_articles).keys() if Counter(all_articles)[x] == 1]

    term_unique_articles = [[] for z in terms]
    for article in unique_articles:
        for term in terms:
            if article in articles[term]: term_unique_articles[terms.index(term)] += [article]
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

    #for j in range(len(args)):
    #    sub_counter = Counter({k : 1 for k in args[j].keys()})
    #    global_counter += sub_counter

    result = []
    for counter_d in args:
        subresult = {}
        num_words = sum(counter_d.values())
        for key in counter_d.keys():
            subresult[key] = (1.0 * counter_d[key] / num_words) * log(1.0 + 1.0 * len(args) / global_counter[key])

        result.append(subresult)
    return result


def main(section, year, month):

    terms = get_lines("../topics.txt")

    e_terms = []
    if check_unrelevant:
        e_terms = terms + ["uncovered"]
    else:
        e_terms = terms

    print "Getting relevant articles..."
    articles, n_articles = group_articles("../arxiv/{0}/{1}/{2:02d}/".format(section, year, month),
                                          terms, return_unrelevant=check_unrelevant)

    unique_articles = get_unique_articles(articles, e_terms, articles)

    print "Coverage:", round(1.0 * sum([len(articles[x]) for x in articles.keys() if x != "uncovered"]) / n_articles, 2)
    print "Unique coverage:", round(1.0 * (sum([len(x) for x in unique_articles])
                                           - int(check_unrelevant) * len(unique_articles[-1])) / n_articles, 2)

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

    topics_texts = []
    for g in range(len(unique_articles)):
        topic_sentences = []
        for file in unique_articles[g]:
            text = " ".join(line_filter(
                                ascii_normalize(
                                    open(file, "r").readlines()))).split(" ")
            topic_sentences.append(text)
        topics_texts.append(topic_sentences)

    if biGram:
        bigram_transformer = Phrases([sentence for topic_content in topics_texts
                                                for sentence in topic_content])
        topics_texts = [bigram_transformer[topic_content] for topic_content in topics_texts]

    count_base = []
    for topic_content in topics_texts:
        cnt_d = sum(map(Counter, topic_content), Counter())

        if '' in cnt_d.keys(): del cnt_d['']  # python "feature": counting zero string
        count_base.append(dict(cnt_d))

    print "Calculating tf-idf..."
    top = tf_idf(*count_base)

    if not os.path.isdir(stat_path): os.makedirs(stat_path)

    for v in range(len(unique_articles)):

        report = open("{}{}.csv".format(stat_path, e_terms[v]), "w+")
        report.write("sep=,\n")

        print "-" * 22
        print e_terms[v].upper()
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

def arg_run():
    if len(argv) < 2:
        print "Error: too few arguments"
    elif len(argv) > 2:
        print "Error: too many arguments"
    else:
        s, y, m = argv[1].split(".")
        y, m = int(y), int(m)
        main(s, y, m)

if __name__ == "__main__":
    stop_list = get_lines("stoplist.txt")
    arg_run()

