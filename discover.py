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
    # FIXME: prefix = "./"

    print("STAGE #1")
    os.system("{}lada.py {} -s {}".format(prefix, section, postfix))
    print("STAGE #2")
    os.system("{}freq.py {} {}".format(prefix, section, postfix))
    print("STAGE #3")
    os.system("{}cite.py {} {}".format(prefix, section, postfix))
    print("STAGE #4")
    os.system("{}wove.py {} -b {}".format(prefix, section, postfix))
    print("STAGE #5")
    os.system("{}lada.py {} {}".format(prefix, section, postfix))
