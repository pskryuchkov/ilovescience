from sys import argv
import os

if len(argv) < 2:
    print "Error: too few arguments"
elif len(argv) > 3:
    print "Error: too many arguments"
else:
    section = argv[1]

    os.system("python lada.py {} -s".format(section))
    os.system("python freq.py {}".format(section))
    os.system("python cite.py {}".format(section))
    os.system("python wove.py {} -b".format(section))
    os.system("python lada.py {}".format(section))
