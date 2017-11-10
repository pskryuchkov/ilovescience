from nbopen import nbopen
from sys import argv
from os.path import isfile
import json

if len(argv) == 2:
    section = argv[1]

    if not isfile(section):
        data = json.load(open('template.json'))
        text_data = json.dumps(data).replace("{section}", section)

        with open("{0}.ipynb".format(section), "w") as f:
            f.write(text_data)

    nbopen.nbopen("{}.ipynb".format(section))
