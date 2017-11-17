from gensim.models import LdaMulticore, Phrases
from unidecode import unidecode
from gensim import corpora
from glob import glob
from sys import argv
import logging
import pickle
import random
import errno
import os
import re


stat_path = "../stat/lda/"

class Volume:
    n_articles = 1000

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
    for g, file in enumerate(file_list[:n_articles]):
        print "{}/{} {}".format(g + 1, n_articles, fn_pure(file))
        text = " ".join(line_filter(
                            ascii_normalize(
                                open(file, "r").readlines()))).lower().split(".")

        base += [x.split() for x in text]
    return base


def calculate_keys(vol, n_top, n_pass, cache_corpus=False,
                   cache_model=False):
    texts_path = "../arxiv/{0}/{1}/{2}/".format(vol.section,
                                                     vol.year, vol.month)

    if not os.path.isdir(texts_path):
        raise Exception('There is no such path: {}'.format(texts_path))

    texts = prepare_sentences(glob(texts_path + "*.txt"), vol.n_articles)

    print("Searching for bigrams...")
    bigram_transformer = Phrases(texts, min_count=10)

    texts = bigram_transformer[texts]

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
        with open(stat_path + "{0}.{1}.{2}.corpus".format(vol.section,
                                                       vol.year, vol.month), 'wb') as f:
            pickle.dump(corpus, f)

    if cache_model:
        lda.save(stat_path + "{0}.{1}.{2}.{3}.lda".format(vol.section,
                                                          vol.year, vol.month, n_pass))
    return lda


def topics(arxiv, n_top=30, n_pass=30, short_keylist=True, choice="r"):
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
                if choice == "r":
                    choice = random.choice(topic)
                elif choice == "f":
                    choice = topic[0]

                f.write("{}\n".format(choice[1]))
    else:
        report = open(stat_path + "{0}.{1}.{2}.keys.csv".format(arxiv.section,
                                                           arxiv.year, arxiv.month), "w+")
        report.write("sep=,\n")

        for c, topic in enumerate(table):
            report.write("topic #{}\n".format(c + 1))

            for word in topic:
                report.write("{0},{1}\n".format(word[0], word[1]))

        report.close()


def arg_run():
    if len(argv) < 2:
        print "Error: too few arguments"
    elif len(argv) > 3:
        print "Error: too many arguments"
    else:
        section, s_year, s_month = argv[1].split(".")
        year, month = int(s_year), int(s_month)
        n_topics = 30
        n_passes = 30

        short_flag = False
        if "-s" in argv:
            short_flag = True
            n_topics = 10
            n_passes = 15

        arxiv_vol = Volume(section, year, month)
        create_dir(stat_path)

        # optimal values: n_topics and n_passes ~ 30
        topics(arxiv_vol, n_topics, n_passes, short_keylist=short_flag)

if __name__ == "__main__":
    stoplist = [x.rstrip() for x in open("extra/stoplist.txt", "r").readlines()]
    arg_run()
