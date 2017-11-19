# Keywords extracting (latent dirichlet allocation used)
# Usage: 'python lada.py cond-mat.16.03'
# FIXME usage: './lada.py cond-mat.17'

# Tested for Anaconda Python 2.7.13
# If script doesn't work, check your Python interpteter version.

from gensim.models import LdaMulticore, Phrases
from unidecode import unidecode
from gensim import corpora
from sys import argv
from os import chdir
import os.path
import logging
import fnmatch
import pickle
import random
import errno
import os
import re


n_articles = 1000
n_articles_debug = 100

stat_path = "../stat/lda/"
volume = None


class Volume:
    def __init__(self, section, year, month):
        self.section = section
        self.year = str(year).zfill(2)
        self.month = str(month).zfill(2)


def ascii_normalize(text):
    # .decode("ascii", "ignore")
    return [unidecode(line.decode("utf-8")) for line in text]


def create_dir(dn):
    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def line_filter(text, min_length=4):
    brackets = re.compile(r'{.*}')  # remove formulas
    alphanum = re.compile(r'[\W_]+')

    filtered = []
    for line in text:
        nline = brackets.sub(' ', line).strip()
        nline = alphanum.sub(' ', nline).strip()
        nline = " ".join([x for x in nline.split()
                          if len(x) >= min_length  # FIXME: empty strings
                          and not x.isdigit()
                          and x not in stoplist])

        filtered.append(nline.lower())

    return filtered


def fn_pure(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def prepare_sentences(file_list, n_articles):
    base = []
    d = {}

    if not os.path.isfile('extra/{}.cache'.format(volume)):
        for g, file in enumerate(file_list[:n_articles]):
            print "{}/{} {}".format(g + 1, n_articles, fn_pure(file))
            text = " ".join(line_filter(
                                ascii_normalize(
                                    open(file, "r").readlines()))).lower().split(".")
            d[file] = text
            base += [x.split() for x in text]

        with open('extra/{}.cache'.format(volume), 'wb') as f:
            pickle.dump(d, f)
    else:
        with open('extra/{}.cache'.format(volume), 'rb') as f:
            d = pickle.load(f)

        for g, file in enumerate(d.keys()):
            print "{}/{} {}".format(g + 1, n_articles, fn_pure(file))
            text = d[file]

            base += [x.split() for x in text]
    return base


def random_glob(path, n_files, mask="*.txt"):
    file_list = []

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            file_list.append(os.path.join(root, filename))

    random.shuffle(file_list)
    return file_list[:n_files]


def calculate_keys(vol, n_top, n_pass, cache_corpus=False,
                   cache_model=False):

    texts_path = "../arxiv/{0}/{1}/".format(vol.section, vol.year)

    if not os.path.isdir(texts_path):
        raise Exception('There is no such path: {}'.format(texts_path))

    files_list = random_glob(texts_path, n_articles)

    texts = prepare_sentences(files_list, n_articles)

    print("Searching for bigrams...")

    # FIXME: bigram transformer returns empty list 
    #bigram_transformer = Phrases(texts, min_count=10)
    #texts = bigram_transformer[texts]

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
        with open(stat_path + "{0}.corpus".format(volume), 'wb') as f:
            pickle.dump(corpus, f)

    if cache_model:
        lda.save(stat_path + "{0}.{1}.lda".format(volume, n_pass))
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
        report = open(stat_path + "{0}.keys.csv".format(volume), "w+")
        report.write("sep=,\n")

        for c, topic in enumerate(table):
            report.write("topic #{}\n".format(c + 1))

            for word in topic:
                report.write("{0},{1}\n".format(word[0], word[1]))

        report.close()


def arg_run():
    if len(argv) < 2:
        print "Error: too few arguments"
    elif len(argv) > 4:
        print "Error: too many arguments"
    else:
        if "-d" in argv:
            global n_articles
            n_articles = n_articles_debug

        global volume
        volume = argv[1]

        section, year_s = volume.split(".")
        year = int(year_s)
        arxiv_vol = Volume(section, year, 0)

        n_topics = 30
        n_passes = 30

        short_flag = False
        if "-s" in argv:
            short_flag = True
            n_topics = 10
            n_passes = 15

        create_dir(stat_path)

        # optimal values: n_topics and n_passes ~ 30
        topics(arxiv_vol, n_topics, n_passes, short_keylist=short_flag)

if __name__ == "__main__":
    chdir(os.path.dirname(os.path.realpath(__file__)))
    stoplist = [x.rstrip() for x in open("extra/stoplist.txt", "r").readlines()]
    arg_run()
