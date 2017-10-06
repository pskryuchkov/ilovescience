# texts frequency analysis
# for each term statistics save to '../stat/frequency/<term>.csv'
# usage: 'python freq.py cond-mat.16.03'
from collections import defaultdict, Counter
from gensim.models import Phrases
from unidecode import unidecode
from glob import glob
from sys import argv
import numpy as np
import re
import os


biGram = True
n_results = 10
n_articles = 100
check_unrelevant = True
stat_path = "../stat/frequency/"
section_path = None


class Table:
    def __init__(self, name, content,
                 labels=None):

        self.name = name
        self.content = content
        self.labels = labels

    def sort(self, col_idx, reverse=True):
        sorted_table = []

        for sheet in self.content:
            sorted_table.append(sorted(sheet,
                                       key=lambda x: x[col_idx],
                                       reverse=reverse))

        return Table(self.name, sorted_table, self.labels)


def save_csv(path="./", sep=","):
    def decorator(func):
        def inner(labels, count_dicts):

            table = func(labels, count_dicts)
            for label, sheet in zip(table.labels, table.content):

                f = open(path + label + ".csv", "w")
                f.write("sep={}\n".format(sep))

                for line in sheet:
                    f.write(sep.join(map(lambda x: str(x), line)) + "\n")

                f.close()

            return table
        return inner
    return decorator


def save_excel(path="./"):
    def decorator(func):
        def inner(labels, count_dicts):

            from openpyxl import Workbook
            table = func(labels, count_dicts)

            wb = Workbook(write_only=False)

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
            print path
            wb.save(path + table.name + ".xlsx")
            wb.close()

            return table
        return inner
    return decorator


def console_table(n_print=6, n_sep=30):
    def decorator(func):
        def inner(labels, count_dicts):

            table = func(labels, count_dicts)
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
    return [unidecode(line.decode("utf-8")) for line in text]


def group_articles(path, terms_l, min_cnt=5,
                   return_unrelevant=False):

    articles_d = defaultdict(list)

    for i, file in enumerate(glob(path+"*.txt")[:n_articles]):
        is_relevant = False
        text = open(file).readlines()

        for term in terms_l:

            if file_counter(text, term) >= min_cnt:

                if not is_relevant: is_relevant = True
                articles_d[term].append(file)

        if return_unrelevant and not is_relevant:
            articles_d["uncovered"].append(file)

    return articles_d, len(glob(path+"*.txt"))


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


@save_excel()
def calc_stat(labels, count_dicts):

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

    return Table("topics", sat_base, labels).sort(col_idx=1)


def save_corr_matrix(fn, data):
    keys = set([])

    for x in data:
        keys.add(x[0])
        keys.add(x[1])

    n = len(keys)

    d = dict(zip(list(keys), range(n)))

    corr = np.identity(n) * 1.0

    for rec in data:
        x = d[rec[0]]
        y = d[rec[1]]
        corr[y][x] = rec[2]
        corr[x][y] = rec[2]

    inv_d = {v: k for k, v in d.iteritems()}

    with open(fn, "w") as table:
        table.write("sep=,\n")

        table.write(",")

        for j, key in enumerate(inv_d.keys()):

            if key < n - 1:
                table.write("{},".format(inv_d[key]))
            else:
                table.write("{}".format(inv_d[key]))

        table.write("\n")

        for y in range(n):
            table.write("{},".format(inv_d[y]))

            for x in range(n):

                if x < n - 1:
                    table.write("{},".format(round(corr[x, y], 3)))
                else:
                    table.write("{}".format(round(corr[x, y], 3)))

            table.write("\n")


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def main(section, year, month): # FIXME: too large function

    terms = get_lines("../topics.txt")

    dest_path = stat_path + "{}.{}.{}/".format(section,
                                              str(year).zfill(2),
                                              str(month).zfill(2))
    check_dir(dest_path)

    if check_unrelevant:
        e_terms = terms + ["uncovered"]
    else:
        e_terms = terms

    print "Getting relevant articles..."
    articles, n_articles = group_articles("../arxiv/{0}/{1}/{2:02d}/"
                                          .format(section, year, month),
                                          terms, return_unrelevant=check_unrelevant)

    unique_articles = get_unique_articles(articles, e_terms, articles)

    print "Coverage:", round(1.0 * sum([len(articles[x]) for x in articles.keys()
                                        if x != "uncovered"]) / n_articles, 2)

    print "Unique coverage:", round(1.0 * (sum([len(x) for x in unique_articles])
                                           - int(check_unrelevant)
                                           * len(unique_articles[-1])) / n_articles, 2)

    with open("{}terms_unique.csv".format(dest_path), "w") as file:

        file.write("sep=,\n")
        file.write("term,n,unique\n")

        for p in range(len(terms)):
            file.write("{},{},{}\n".format(terms[p],
                                           len(unique_articles[p]), \

                                           round(1.0 * len(unique_articles[p]) /
                                           len(articles[terms[p]]), 2)))

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

    save_corr_matrix("{}terms_similar.csv".format(dest_path), corr_vals)

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
        print("Searching for bigrams...")
        bigram_transformer = Phrases([sentence for topic_content in topics_texts
                                                for sentence in topic_content])

        topics_texts = [bigram_transformer[topic_content]
                        for topic_content in topics_texts]

    print "Counting words..."
    count_base = []
    for topic_content in topics_texts:
        cnt_d = sum(map(Counter, topic_content), Counter())

        # python "feature": counting zero string
        if '' in cnt_d.keys():
            del cnt_d['']

        count_base.append(dict(cnt_d))

    print "Calculating tf-idf..."

    calc_stat(e_terms, count_base)


def arg_run():
    if len(argv) < 2:
        print "Error: too few arguments"
    elif len(argv) > 2:
        print "Error: too many arguments"
    else:
        global section_path
        section_path = stat_path + argv[1] + "/"

        s, y, m = argv[1].split(".")
        y, m = int(y), int(m)
        main(s, y, m)

if __name__ == "__main__":
    stop_list = get_lines("../stoplist.txt")
    arg_run()

