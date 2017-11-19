from nbopen import nbopen
from sys import argv
from os.path import isfile
import json

if len(argv) == 2:
    section, year = argv[1].split(".")

    if not isfile(section):
        data = json.load(open('notes/template.json'))
        text_data = json.dumps(data)
        
        text_data = text_data.replace("{section}", section)
        text_data = text_data.replace("{year}", year.lstrip("0"))

        with open("notes/{}.{}.ipynb".format(section, year), "w") as f:
            f.write(text_data)

    nbopen.nbopen("notes/{}.{}.ipynb".format(section, year))
