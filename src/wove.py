# Word2vec algorithm applying to scientific texts

# Calculating word vectors
# python wove.py cond-mat.16.03 -b

# Topics word vectors info
# python wove.py cond-mat.16.03 -t

# Word2vec console (pre-calclulated wordvectors required)
# python wove.py cond-mat.16.03

# FIXME usage: './wove.py cond-mat.17'

# Tested for Anaconda Python 2.7.13
# If script doesn't work, check your Python interpteter version.

from gensim.models import Word2Vec, Phrases
from unidecode import unidecode
from sys import argv
from os import chdir
import fnmatch
import logging
import pickle
import random
import errno
import os
import re

min_count = 10
size = 300
window = 10

n_articles = 1000
n_articles_debug = 100
n_ass = 10

# FIXME
stat_path = "../stat/word_vec/"
texts_path = "../arxiv/{0}/{1}/{2:02d}/*.txt"
vec_path = stat_path + "{0}"
topics_path = ""

volume = None

def create_dir(dn):
    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def ascii_normalize(text):
    # .decode("ascii", "ignore")
    return [unidecode(line.decode("utf-8")) for line in text]


# do we need it?
def break_remove(raw_text):
    text = " ".join(raw_text)
    break_expr = re.compile(r'(?![A-Za-z0-9]+)\s*-\s*(?=[A-Za-z0-9]+)')
    text = break_expr.sub("", text)
    return text.split("\n")

# FIXME: plural
def line_filter(text, min_length=3):  # FIXME
    filtered = []
    exp1 = re.compile(r'==')
    exp2 = re.compile(r'{.*}')
    # exp3 = re.compile(r'^[0-9]+')
    alphanum = re.compile(r'[\W_]+')
    for line in text:
        nline = alphanum.sub(' ', line).strip()
        if len(exp1.findall(nline)) == 0 \
                and len(exp2.findall(nline)) == 0:

            # sline = map(lambda x: normalizer.stem(x), nline.split())
            nline = " ".join([x for x in nline.split()
                                if len(x) >= min_length
                                and not x.isdigit()
                                and not x.lower() in stoplist])

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


def build_word_vec(show_log=True):

    section, year = volume.split(".")
    texts_path = "../arxiv/{0}/{1}/".format(section, year)

    files_list = random_glob(texts_path, n_articles)
    sentences = prepare_sentences(files_list, n_articles)

    if show_log:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)

    # FIXME: bigram transformer returns empty list
    #bigram_transformer = Phrases(sentences)
    #return Word2Vec(bigram_transformer[sentences], min_count=min_count, size=size, window=window, workers=4)

    return Word2Vec(sentences, min_count=min_count, size=size, window=window, workers=4)

def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def replics(model, target_word, topn=10):
    return {word: model.vocab[word].count
                           for word in model.vocab.keys()
                           if word.find(target_word) == 0}


def topics_interset(model):
    topics = get_lines(topics_path)

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
        print "{} {} ({})".format(topics_normalized[pair[0]], topics_normalized[pair[1]], cnt)
        for x in sorted(similar, key=(lambda x: x[1]), reverse=True):
            if x[1] >= 0.6:
                print "  {} {}".format(x[0], round(x[1],2))


def topics_normalize(model, raw_topics):
    norm_topics = []
    for word in raw_topics:
        r = replics(model, word)
        if r is not None:
            print "{} -> {} ({})".format(word, max(r, key=r.get), r[max(r, key=r.get)])
            norm_topics.append(max(r, key=r.get))
        else:
            norm_topics.append(None)
            print word, "None"

    return norm_topics


def topics_info(word_vec_path):
    model = Word2Vec.load(word_vec_path)

    topics = topics_normalize(model, get_lines(topics_path))

    for word in topics:
        if word in model.vocab.keys():
            print word
            for neighbor in model.most_similar(positive=[word], topn=n_ass):
                print "  '{}', {}, {}".format(neighbor[0], round(neighbor[1], 2),
                                              model.vocab[neighbor[0]].count)
        else:
            print word, "None"


def vocab(word_vec_path):
    model = Word2Vec.load(word_vec_path)
    keys = model.vocab.keys()
    keys.sort()
    return keys


def console(path):
    model = Word2Vec.load(path)

    print "Vocabulary length: {} words".format(len(model.vocab.keys()))
    print "Minimal count: {} words".format(model.min_count)

    while True:
        query = raw_input('> ')
        if query in model.vocab.keys():
            for record in model.most_similar(positive=[query], topn=10):
                print "{}, {}, {}".format(record[0], round(record[1], 2), model.vocab[record[0]].count)
        else:
            print "There is no such word: '{}'".format(query)


def arg_run():
    if len(argv) < 2:
        raise Exception("Too few arguments")

    if len(argv) > 4:
        raise Exception("Too many arguments")

    if "-d" in argv:
        global n_articles
        n_articles = n_articles_debug

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
        print "Word2vec on arxiv {}.{:02d}".format(y)
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
    # initialization
    chdir(os.path.dirname(os.path.realpath(__file__)))
    stoplist = get_lines("extra/stoplist.txt")
    create_dir(stat_path)

    arg_run()

# TODO: discover relations
#print model.doesnt_match("insulating superconductive d-wave cdw sdw paramagnetic".split())

#pprint(model.most_similar(positive=["graphene", "experiment"], negative=["theory"], topn=n_ass))
#pprint(model.most_similar(positive=["graphene", "theory"], negative=["experiment"], topn=n_ass))

#pprint(model.most_similar(positive=["superconducting", "low_temperature"], negative=["high_temperature"], topn=n_ass))
#pprint(model.most_similar(positive=["superconducting", "high_temperature"], negative=["low_temperature"], topn=100))

#pprint(model.most_similar(positive=["quantum", "macroscopic"], negative=["microscopic"], topn=100)) # qubits bose-einstein

#pprint(model.most_similar(positive=["gas", "cold"], negative=["heat"], topn=n_ass))
