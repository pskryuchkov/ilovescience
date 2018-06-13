from sys import argv
import os

if len(argv) < 2:
    print "Error: too few arguments"
elif len(argv) > 3:
    print "Error: too many arguments"
else:
    section = argv[1]
    prefix = "python -W ignore src/"
    postfix = ""
    # postfix = ""
    # FIXME: prefix = "./"

    print("STAGE #1")
    os.system("{}lda.py {} -s {}".format(prefix, section, postfix))
    print("STAGE #2")
    os.system("{}terms_cn.py {} {}".format(prefix, section, postfix))
    print("STAGE #3")
    os.system("{}cites.py {} {}".format(prefix, section, postfix))
    print("STAGE #4")
    os.system("{}word2vec.py {} -b {}".format(prefix, section, postfix))
    print("STAGE #5")
    os.system("{}lda.py {} {}".format(prefix, section, postfix))
