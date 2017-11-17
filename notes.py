from nbopen import nbopen
from sys import argv
from os.path import isfile
import json

if len(argv) == 2:
    section, year, month = argv[1].split(".")

    if not isfile(section):
        data = json.load(open('notes/template.json'))
        text_data = json.dumps(data)
        
        text_data = text_data.replace("{section}", section)
        text_data = text_data.replace("{year}", year.lstrip("0"))
        text_data = text_data.replace("{month}", month.lstrip("0"))

        with open("notes/{}.{}.{}.ipynb".format(section, year, month), "w") as f:
            f.write(text_data)

    nbopen.nbopen("notes/{}.{}.{}.ipynb".format(section, year, month))
