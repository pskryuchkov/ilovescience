# Downloading articles from arxiv.org.
# PDF files after loading converted to txt (on fly)
# Usage: './article_crawl.py -s cond-mat -y 16 -m 2'

# Articles saved on '../arxiv/<section>/<year>/<month>'

import logging
from io import BytesIO
from urllib.request import urlopen

from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine

from os.path import isfile, join, splitext, isdir
from os import listdir, makedirs, chdir
import subprocess
import signal
import sys
import os

import argparse
import time
import re

year, month, section = None, None, None
index_path = None
target_url = None
time_lim = 20

# cn.arXiv.org (China, OK)
# de.arXiv.org (Germany, Bad)
# in.arXiv.org (India, Bad)
# es.arXiv.org (Spain, OK)
# lanl.arXiv.org (U.S. mirror at Los Alamos, OK)

valid_mirrors = ["cn.", "de.", "in.", "es.", "lanl."]
mirror = ""


def pdf_to_txt(url):
    fp = BytesIO(urlopen(url).read())
    parser = PDFParser(fp)
    doc = PDFDocument()
    parser.set_document(doc)
    doc.set_parser(parser)
    doc.initialize('')
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()

    for param in ("all_texts", "detect_vertical", "word_margin",
                  "char_margin", "line_margin", "boxes_flow"):
        paramv = locals().get(param, None)
        if paramv is not None:
            setattr(laparams, param, paramv)

    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    extracted_text = ''
    return_code = 0
    for page in doc.get_pages():
        try:
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    extracted_text += lt_obj.get_text()
        except:
            print("PDF converting error: {0}".format(url))
            return_code = 1
            break

    if return_code == 0:
        print("convert ok")

    return return_code, extracted_text


def main():
    signal.signal(signal.SIGALRM, handler) # func timeout set

    if 'darwin' in sys.platform:
        subprocess.Popen('caffeinate') # stay active (os x)

    print("Connecting to arxiv...")
    response = urlopen(target_url).read().decode("utf-8")
    articles = list(set(re.findall('arXiv:({0}{1}.[0-9]+)'.format(year, str(month).zfill(2)), response)))
    n = len(articles)

    base = []
    if isdir(index_path):
        base = [splitext(f)[0] for f in listdir(index_path) if isfile(join(index_path, f)) and
                                        splitext(f)[1] == ".txt"] # FIXME
    else:
        makedirs(index_path)

    print("Loading articles...")
    for j in range(len(articles)):
        signal.alarm(time_lim)
        try:
            if articles[j] not in base:
                link = "http://{0}arXiv.org/pdf/{1}.pdf".format(mirror, articles[j])

                init_time = time.time()
                code, text = pdf_to_txt(link)
                if code == 0:
                    print("{0}/{1}: {2}, {3}s".format(j + 1, n, articles[j],
                                                      round(time.time() - init_time, 2)))

                    with open('{0}/{1}.txt'.format(index_path, articles[j]), 'a') as the_file:
                        the_file.write(text)
            else:
                print("{0} already loaded".format(articles[j]))

            signal.alarm(0)

        except Exception as exc:
            print("{}: {}".format(articles[j], exc))


def handler(signum, frame):
    raise Exception("Timeout")


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
        print("Error: too few arguments")
        exit()

    section = args.s[0]
    year = int(args.y[0])
    month = int(args.m[0])

    if args.o:
        mirror = args.o[0] + "."
        if mirror in valid_mirrors:
            print("Mirror: {0}".format(mirror[:-1]))
        else:
            print("Invalid mirror: {}".format(mirror))
            exit()

    print("Section: {}, year: {}, month: {}".format(section, year, month))

    index_path = "../arxiv/{0}/{1}/{2}".format(section, year, str(month).zfill(2))

    target_url = "https://arxiv.org/list/{0}/{1}{2}?show=10000". \
        format(section, year, "%02d" % (month,))

    main()


if __name__ == "__main__":
    chdir(os.path.dirname(os.path.realpath(__file__)))
    logging.getLogger().setLevel(logging.ERROR)
    logging.propagate = False
    arg_run()
