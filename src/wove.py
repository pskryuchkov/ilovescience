# word2vec algorithm applying to scientific articles base
#
# calculate word vectors
# python wotvec.py -b cond-mat.16.03
#
# topics word vectors info
# python wotvec.py -t cond-mat.16.03
#
# word2vec console
# python wotvec.py -c cond-mat.16.03

from gensim.models import Word2Vec, Phrases
from unidecode import unidecode
from glob import glob
from sys import argv
import argparse
import logging
import re

min_count = 10
size = 300
window = 10
n_articles = 2000
n_ass = 10

texts_path = "../arxiv/{0}/{1}/{2:02d}/*.txt"
vec_path = "../stat/word_vec/{0}.{1}.{2:02d}.wordvec"
topics_path = "../topics.txt"


def ascii_normalize(text):
    # .decode("ascii", "ignore")
    return [unidecode(line.decode("utf-8")) for line in text]


# do we need it?
def break_remove(raw_text):
    text = " ".join(raw_text)
    break_expr = re.compile(r'(?![A-Za-z0-9]+)\s*-\s*(?=[A-Za-z0-9]+)')
    text = break_expr.sub("", text)
    return text.split("\n")


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
                                and not x in stoplist])

            filtered.append(nline.lower())

    return filtered


def prepare_sentences(file_list, n_articles=n_articles):
    base = []
    for g, file in enumerate(file_list[:n_articles]):
        print file
        text = " ".join(line_filter(
                            ascii_normalize(
                                open(file, "r").readlines()))).lower().split(".")

        base += [x.split() for x in text]
    return base


def build_word_vec(path, show_log = True):
    sentences = prepare_sentences(glob(path))
    if show_log:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
    bigram_transformer = Phrases(sentences)
    return Word2Vec(bigram_transformer[sentences], min_count=min_count, size=size, window=window, workers=4)


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
            print "There is not such word: '{}'".format(query)


def arg_run():
    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-b', nargs=1, help="Build Wordvec")
    parser.add_argument('-t', nargs=1, help="Topics Info")
    parser.add_argument('-c', nargs=1, help="Console")

    args = parser.parse_args()

    arg_sum = sum(map(int, [args.b is not None,
                       args.t is not None,
                       args.c is not None]))

    if arg_sum < 1:
        print "Error: too few arguments"
    elif arg_sum > 1:
        print "Error: too many arguments"
    elif arg_sum == 1:
        s, y, m = argv[2].split(".")
        y, m = int(y), int(m)
        if args.b is not None:
            # build
            model = build_word_vec(texts_path.format(s, y, m))
            model.save(vec_path.format(s, y, m))
        elif args.t is not None:
            # topic info
            topics_info(vec_path.format(s, y, m))
        else:
            # console
            print "Word2vec on arxiv {}.{:02d}".format(y, m)
            console(vec_path.format(s, y, m))

if __name__ == "__main__":
    # initialization
    stoplist = get_lines("stoplist.txt")
    arg_run()

# TODO: discover relations
#print model.doesnt_match("insulating superconductive d-wave cdw sdw paramagnetic".split())

#pprint(model.most_similar(positive=["graphene", "experiment"], negative=["theory"], topn=n_ass))
#pprint(model.most_similar(positive=["graphene", "theory"], negative=["experiment"], topn=n_ass))

#pprint(model.most_similar(positive=["superconducting", "low_temperature"], negative=["high_temperature"], topn=n_ass))
#pprint(model.most_similar(positive=["superconducting", "high_temperature"], negative=["low_temperature"], topn=100))

#pprint(model.most_similar(positive=["quantum", "macroscopic"], negative=["microscopic"], topn=100)) # qubits bose-einstein

#pprint(model.most_similar(positive=["gas", "cold"], negative=["heat"], topn=n_ass))


