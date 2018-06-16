from sys import argv, path
import itertools
import pickle
import os

path.insert(0, os.path.dirname(
                    os.path.realpath(__file__)) + '/extra')
import shared
import config


def prepare_sentences(file_list, n_articles, volume):
    base = []
    d = {}

    if not os.path.isfile('cache/{}.cache'.format(volume)):
        for g, file in enumerate(file_list[:n_articles]):

            print("{}/{} {}".format(g + 1, n_articles, shared.fn_pure(file)))

            text = " ".join(shared.line_filter(
                                shared.ascii_normalize(
                                    open(file, "r").readlines()), min_length=3)).lower().split(".")

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


def count_terms(section, year, n_articles=10000):
    result_path = "../stat/freq"
    volume = "{}.{}".format(section, year)
    texts_path = "../arxiv/{0}/{1}/".format(section, year)
    
    keys = [line.strip() for line in open("../topics/{}.txt".format(section), "r").readlines()]
    files = shared.random_glob(texts_path, n_articles)
    texts = prepare_sentences(files, n_articles, volume)

    cnt = []
    texts_union = list(itertools.chain.from_iterable(texts))
    for h, k in enumerate(keys):
        print(k)
        cnt.append([k, texts_union.count(k) / len(texts)])

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open("{}/{}.{}.csv".format(result_path, section, int(year)), "w") as report:
        report.write("sep=;\n")
        for line in cnt:
            str_line = map(str, line)
            report.write(";".join(str_line) + "\n")

def arg_run():
    if len(argv) < 2:
        print("Error: too few arguments")
    elif len(argv) > 3:
        print("Error: too many arguments")
    else:
        section, year = argv[1].split(".")
        year = int(year)

        if "-d" in argv:
            count_terms(section, year, config.n_articles_debug)
        else:
            count_terms(section, year)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    shared.create_dir(config.terms_stat)
    arg_run()
    #count_terms("cond-mat", 10)