from unidecode import unidecode
import fnmatch
import random
import errno
import os
import re


def ascii_normalize(text):
    return [unidecode(line.decode("utf-8")) for line in text]


def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def fn_pure(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def create_dir(dn):
    if not os.path.exists(dn):
        try:
            os.makedirs(dn)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def random_glob(path, n_files, mask="*.txt"):
    file_list = []

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, mask):
            file_list.append(os.path.join(root, filename))

    random.shuffle(file_list)
    return file_list[:n_files]


class SingleTable:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def sort(self, col_idx, reverse=True):
        sorted_table = sorted(self.content,
                              key=lambda x: x[col_idx],
                              reverse=reverse)

        return SingleTable(self.name, sorted_table)

    def to_multi(self):
        return MultiTable(self.name, [self.content], [self.name])


class MultiTable:
    def __init__(self, name, content,
                 labels):

        self.name = name
        self.labels = labels
        self.content = content

    def sort(self, col_idx, reverse=True):
        sorted_table = []

        for sheet in self.content:
            sorted_table.append(sorted(sheet,
                                       key=lambda x: x[col_idx],
                                       reverse=reverse))

        return MultiTable(self.name, sorted_table, self.labels)


def save_csv(path="", sep=","):
    def decorator(func):
        def inner(*args):

            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            for label, sheet in zip(table.labels, table.content):

                f = open(path + label + ".csv", "w")
                f.write("sep={}\n".format(sep))

                for line in sheet:
                    f.write(sep.join(map(lambda x: str(x), line)) + "\n")

                f.close()

            return table
        return inner
    return decorator


def save_excel(path=""):
    def decorator(func):
        def inner(*args):

            from openpyxl import Workbook
            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            wb = Workbook()

            first_sheet = True
            for label, sheet in zip(table.labels, table.content):

                if first_sheet:
                    first_sheet = False
                    ws = wb.active
                    ws.title = label
                else:
                    ws = wb.create_sheet(label)

                for line in sheet:
                    ws.append(line)

            wb.save(path + table.name + ".xlsx")
            wb.close()

            return table
        return inner
    return decorator


def console_table(n_print=6, n_sep=30):
    def decorator(func):
        def inner(*args):

            table = func(*args)

            if isinstance(table, SingleTable):
                table = table.to_multi()

            for label, sheet in zip(table.labels, table.content):

                print("-" * n_sep)
                print(label.upper())
                print("-" * n_sep)

                for j, line in enumerate(sheet):
                    if j < n_print:
                        print(", ".join(list(map(lambda x: str(x), line))))
                    else:
                        print("...")
                        break

            return table
        return inner
    return decorator


# FIXME: empty strings, plural
def line_filter(text, min_length):
    brackets = re.compile(r'{.*}') # formulas
    alphanum = re.compile(r'[\W_]+')

    filtered = []
    for line in text:
        nline = brackets.sub(' ', line).strip()
        nline = alphanum.sub(' ', nline).strip()

        nline = " ".join([x for x in nline.split()
                                if len(x) >= min_length
                                and not x.isdigit()
                                and not x.lower() in stop_list])

        filtered.append(nline.lower())

    return filtered


def plural_filter(texts):
    f_base = []

    flat = set([x for sublist in texts for x in sublist])

    for j, text in enumerate(texts):
        f_text = []
        for word in text:
            if word[-1] == "s" and not word[-2] == "s":
                candidate = word[:-1]
                if candidate in flat:
                    f_text.append(candidate)
                else:
                    f_text.append(word)
            else:
                f_text.append(word)

        f_base.append(f_text)

    return f_base


stop_list = get_lines(os.path.dirname(os.path.realpath(__file__)) + "/stoplist.txt")