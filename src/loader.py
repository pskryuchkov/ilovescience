# Getting articles from arxiv.org.
# PDF files are loaded and converted to txt on fly
# Example of usage: './loader.py -s cond-mat -y 16 -m 2'
# Articles saved on '../arxiv/<section>/<year>/<month>'
# Tested for Anaconda Python 2.7.13
# If script doesn't work, check your Python interpteter version.


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

from os.path import isfile, join, splitext, isdir
from os import listdir, makedirs
import subprocess
import urllib2
import signal
import sys

import argparse
import time
import re


year = None
month = None
section = None
index_path = None
target_url = None

maxArticles = 2000
time_lim = 30


def pdf_from_url_to_txt(url):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()

    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())

    start_loading = time.time()
    fp = StringIO(urllib2.urlopen(url).read())
    loading_time = time.time() - start_loading

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pagenos=set()

    start_processing = time.time()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=0, password="", caching=True,
                                  check_extractable=True):
        try:
            interpreter.process_page(page)
        except:
            print "PDF converting error: {0}".format(url)

    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    processing_time = time.time() - start_processing
    return text, loading_time, processing_time


def arg_run():
    global year, month, section, \
        index_path, target_url

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-s', nargs=1, help="Arxiv section")
    parser.add_argument('-y', nargs=1, help="Year")
    parser.add_argument('-m', nargs=1, help="Month")
    args = parser.parse_args()

    if args.s is None \
            or args.y is None \
            or args.m is None:
        print "Error: too few arguments"
        exit()

    section = args.s[0]
    year = int(args.y[0])
    month = int(args.m[0])
    print "Section: {}, year: {}, month: {}".format(section, year, month)

    index_path = "../arxiv/{0}/{1}/{2}".format(section, year, str(month).zfill(2))

    target_url = "https://arxiv.org/list/{0}/{1}{2}?show=10000". \
        format(section, year, "%02d" % (month,))

    main()


def main():
    signal.signal(signal.SIGALRM, handler) # func timeout set
    signal.alarm(time_lim)
    if 'darwin' in sys.platform:
        subprocess.Popen('caffeinate') # stay active (os x)

    print "Connecting to arxiv..."
    data = urllib2.urlopen(target_url)

    articles = []
    for line in data:
        result = re.findall('arXiv:({0}{1}.[0-9]+)'.format(year,
                                                    str(month).zfill(2)), line)
        if result != []:
            articles.append(result[0])

    articles = list(set(articles))
    n = len(articles)

    base = []
    if isdir(index_path):
        base = [splitext(f)[0] for f in listdir(index_path) if isfile(join(index_path, f)) and
                                        splitext(f)[1] == ".txt"] # FIXME
    else:
        makedirs(index_path)

    print "Loading articles..."
    for j in range(min(len(articles), maxArticles)):
        try:
            if articles[j] not in base:
                link = "https://arxiv.org/pdf/{0}.pdf".format(articles[j])

                text, load_time, convert_time = pdf_from_url_to_txt(link)

                print "{0}/{1}: {2}, {3}s / {4}s".format(j + 1,
                                                        n,
                                                        articles[j],
                                                        round(load_time, 2),
                                                        round(convert_time, 2))

                with open('{0}/{1}.txt'.format(index_path, articles[j]), 'a') as the_file:
                    the_file.write(text)
            else:
                print "{0} already loaded".format(articles[j])

        except Exception, exc:
            if str(exc) == "Timeout":
                print "Converting timeout: {0}".format(articles[j])
            else:
                print "{}: {}".format(articles[j], exc)


def handler(signum, frame):
    raise Exception("Timeout")


if __name__ == "__main__":
    arg_run()
