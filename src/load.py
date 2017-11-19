# Downloading articles from arxiv.org.
# PDF files after loading converted to txt (on fly)
# Usage: './load.py -s cond-mat -y 16 -m 2'
# FIXME usage: './load.py cond-mat.17'

# Articles saved on '../arxiv/<section>/<year>/<month>'

# Tested for Anaconda Python 2.7.13
# If script doesn't work, check your Python interpteter version.


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

from os.path import isfile, join, splitext, isdir
from os import listdir, makedirs, chdir
import subprocess
import urllib2
import signal
import sys

import argparse
import time
import re

import socks
import socket

year = None
month = None
section = None
index_path = None
target_url = None

maxArticles = 2000
time_lim = 40

# cn.arXiv.org (China, OK)
# de.arXiv.org (Germany, Bad)
# in.arXiv.org (India, Bad)
# es.arXiv.org (Spain, OK)
# lanl.arXiv.org (U.S. mirror at Los Alamos, OK)

valid_mirrors = ["cn.", "de.", "in.", "es.", "lanl."]
mirror = ""


def pdf_from_url_to_txt(url):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()

    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())

    start_loading = time.time()
    fp = StringIO(urllib2.urlopen(url).read())
    print "load ok"
    loading_time = time.time() - start_loading

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pagenos=set()

    start_processing = time.time()
    err_flag = False
    for page in PDFPage.get_pages(fp, pagenos, maxpages=30, password="", caching=False,
                                  check_extractable=True):
        try:
            interpreter.process_page(page)
        except Exception, exc:
            if str(exc) == "Timeout":
                print "Converting timeout: {0}".format(url)
            else:
                print "PDF converting error: {0}".format(url)
            err_flag = True
            break

    if not err_flag: print "convert ok"
    text = retstr.getvalue()
    fp.close()
    device.close()
    retstr.close()
    processing_time = time.time() - start_processing
    return text, loading_time, processing_time, err_flag


def arg_run():
    global year, month, section, \
        index_path, target_url, mirror

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-s', nargs=1, help="Arxiv section")
    parser.add_argument('-y', nargs=1, help="Year")
    parser.add_argument('-m', nargs=1, help="Month")
    parser.add_argument('-o', nargs=1, help="Origin")
    args = parser.parse_args()

    if args.s is None \
            or args.y is None \
            or args.m is None:
        print "Error: too few arguments"
        exit()

    section = args.s[0]
    year = int(args.y[0])
    month = int(args.m[0])

    if args.o is not None:
        mirror = args.o[0]

    print "Section: {}, year: {}, month: {}".format(section, year, month)

    index_path = "../arxiv/{0}/{1}/{2}".format(section, year, str(month).zfill(2))

    target_url = "https://arxiv.org/list/{0}/{1}{2}?show=10000". \
        format(section, year, "%02d" % (month,))

    main()


def main():
    global mirror
    signal.signal(signal.SIGALRM, handler) # func timeout set

    if 'darwin' in sys.platform:
        subprocess.Popen('caffeinate') # stay active (os x)

    if len(mirror) > 0:
        if mirror in valid_mirrors:
            print "Mirror: {0}".format(mirror)
        elif mirror == "default":
            mirror = ""
        else:
            print "Invalid mirror: {}".format(mirror)
            exit()

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
        signal.alarm(time_lim)
        try:
            if articles[j] not in base:
                link = "http://{0}arXiv.org/pdf/{1}.pdf".format(mirror, articles[j])

                text, load_time, convert_time, err_flag = pdf_from_url_to_txt(link)

                if not err_flag:
                    print "{0}/{1}: {2}, {3}s / {4}s".format(j + 1,
                                                        n,
                                                        articles[j],
                                                        round(load_time, 2),
                                                        round(convert_time, 2))

                    with open('{0}/{1}.txt'.format(index_path, articles[j]), 'a') as the_file:
                        the_file.write(text)
            else:
                print "{0} already loaded".format(articles[j])

            signal.alarm(0)

        except Exception, exc:
            #if str(exc) == "Timeout":
            #    print "Converting timeout: {0}".format(articles[j])
            #else:
            print "{}: {}".format(articles[j], exc)


def handler(signum, frame):
    raise Exception("Timeout")


def create_connection(address, timeout=None, source_address=None):
    sock = socks.socksocket()
    sock.connect(address)
    return sock

socks.socksocket().setproxy(socks.PROXY_TYPE_SOCKS5, "127.0.0.1", 9050)

# patch the socket module
socket.socket = socks.socksocket
socket.create_connection = create_connection

if __name__ == "__main__":
    chdir(os.path.dirname(os.path.realpath(__file__)))
    arg_run()
