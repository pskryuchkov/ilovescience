import urllib.request
from sys import argv
import re
import os


def arg_run():
    if len(argv) < 2:
        print("Error: too few arguments")
    elif len(argv) > 4:
        print("Error: too many arguments")
    else:
        global volume
        volume = argv[1]

        section, year_s = volume.split(".")
        year = int(year_s)

        debug_flag = "-d" in argv

        main(section, year, debug=debug_flag)


def main(section, year, debug=False):
    saving_path = "../arxiv/annotations/{}.{}".format(section, year)

    articles_list_link = 'https://arxiv.org/list/{0}/{1}{2}?show=10000'
    article_link_pattern = 'http://export.arxiv.org/api/query?id_list='
    article_name_pattern = 'arXiv:({0}{1}.[0-9]+)'

    print("Getting links...")
    articles = []
    for month in range(1, 12 + 1):
        print(month)
        target_url = articles_list_link.format(section, str(year), "%02d" % (month,))

        data = urllib.request.urlopen(target_url).read().decode("utf8")

        articles += re.findall(article_name_pattern.format(year, str(month).zfill(2)), data)

        if month > 1 and debug:
            break

    print("Total articles", len(articles))

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    for j, a in enumerate(articles):
        fn = "{}/{}.txt".format(saving_path, a) 

        if os.path.isfile(fn):
            print(fn)
            continue
        else:
            print("{}/{}".format(j+1, len(articles)))
        
        res = urllib.request.urlopen(article_link_pattern + a)\
                    .read().decode("utf8")
        
        with open(fn, "w") as f:
            f.write(res)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    arg_run()