from gensim.models import Word2Vec, LdaMulticore
from IPython.display import display, Markdown
from matplotlib import cm
from numpy import *

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools

import pickle
import os

from warnings import warn
import csv

from bs4 import BeautifulSoup as BS
from collections import Counter
import numpy as np
import matplotlib
from glob import glob
import json

from functools import reduce
import operator
import math
import re


from sys import path
path.insert(0, os.path.dirname(
                    os.path.realpath("__file__")) + '/../src/extra')

import shared

from bs4 import BeautifulSoup as BS
from functools import reduce
from os.path import isfile


annotations_path = "../arxiv/annotations"

cite_table_head = \
"""
| {0}Author(s){1} | {0}Year{1} | {0}Ref. count{1} |
| --- | --- | --- |
""".format("<center>", "<center>")


articles_table_head = \
"""
| Topic title | Articles |
| --- | --- |
"""


def replics(model, target_word, topn=10):
    return {word: model.wv.vocab[word].count
                           for word in model.wv.vocab.keys()
                           if word.find(target_word) == 0}


def topics_normalize(model, raw_topics):
    norm_topics = []
    for word in raw_topics:
        r = replics(model, word)
        if not r:
            norm_topics.append(None)
        else:
            norm_topics.append(max(r, key=r.get))

    return norm_topics


def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def color_convert(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(uint8, array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def add_bar(fig, inp, n_topics, n_keys, n_row, n_col, chart_count):
    if chart_count >= n_topics:
        return

    keys = sorted([x for x in inp[:n_keys]], key=lambda x: x[1])

    labels = [x[1] for x in keys]
    values = [x[0] for x in keys]

    data = go.Bar(
            x=labels,
            y=values,
            orientation = 'h',
            name = "topic #{}".format(chart_count + 1),
            marker = dict(color='orange')
    )
    fig.append_trace(data, (chart_count // n_col) + 1, (chart_count % n_col) + 1)
    fig['layout']['xaxis{0}'.format(chart_count+1)].update(showticklabels = False)
    fig['layout']['yaxis{0}'.format(chart_count+1)].update(ticklen=3)
    fig['layout']['yaxis{0}'.format(chart_count+1)]['tickfont'].update(size=11,
                                                                       color="lightgrey") 


def draw_barchart(volume, n_keys = 6, n_row = 5, n_col = 3):
    fn = "../stat/lda/{}.keys.csv".format(volume)
    n_charts = n_row * n_col

    table = []
    val = []
    key = []
    with open(fn, "rt") as f:
        reader = csv.reader(f)
        next(f)
        for line in reader:
            if len(line) == 1:
                table.append([])
            else:
                table[-1].append([line[1], float(line[0])])

    n_topics = len(table)

    fig = tools.make_subplots(rows=n_row, cols=n_col, print_grid=False,
                              horizontal_spacing = 0.15, vertical_spacing=0.05,
                              subplot_titles=[x[0][0] for x in table[:n_charts]])

    chart_count = 0
    for topic_words in table[:n_charts]:
        add_bar(fig, topic_words, n_topics, n_keys, n_row, n_col, chart_count)
        chart_count += 1

    fig['layout'].update(height=750, width=750, 
                            showlegend=False, title="", 
                            margin=go.Margin(t=120))
    py.iplot(fig, show_link=False);


def citied_articles(volume, n_cites=10, head=cite_table_head):
    markdown_table = head

    with open("../stat/references/{}.txt".format(volume), "r") as f:
        data = f.readlines()

    line_format = "|{0}{2}{1}|{0}{3}{1}|{0}{4}{1}|\n"
    for line in data[:n_cites]:
        authors, year, count = line.split(",")

        markdown_table += line_format.format("<center>", "</center>",
                                              ", ".join(list(map(lambda x: x.strip(), authors.split("&")))),
                                              year.strip(), count.strip())
    display(Markdown(markdown_table))


def fn_pure(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def lda_table(volume):
    fn = "../stat/lda/{}.keys.csv".format(volume)
    table = []
    with open(fn, "rt") as f:
        reader = csv.reader(f)
        next(f)
        for line in reader:
            if len(line) == 1:
                table.append([])
            else:
                table[-1].append([line[1], float(line[0])])
    return table


def topic_occur(text, topic, max_len=5000):
    if len(text) > max_len:
        return 0

    counter = 0
    for word in topic.keys():
        counter += text.count(word) * topic[word]
    return counter


def relevant_articles(volume, n_articles=1000, n_topic_articles=5):
    table = lda_table(volume)
    text_cache = pickle.load(open('../src/cache/{}.cache'.format(volume), 'rb'))

    topics_list = []
    topic_titles = []

    for x in table:
        topic = {}
        for y in x:
            topic[y[0]] = y[1]

        topic_titles.append(", ".join([z[0] for z in x[:2]]))
        topics_list.append(topic)

    text_indexes = list(text_cache.keys())

    score_list = []
    for topic in topics_list:
        occur = []

        for idx in text_indexes[:n_articles]:
            text = text_cache[idx][0].split()

            val = 1.0 * topic_occur(text, topic)
            if val > 0:
                occur.append([fn_pure(idx), val])

        score_list.append(occur)

    sorted_score = [sorted(x, key=lambda z: z[1], reverse=True) for x in score_list]

    relevant = []
    for z in sorted_score:
        sub = []

        for w in z[:n_topic_articles]:
            sub.append(w[0])

        relevant.append(sub)

    return topic_titles, relevant


def lda_print(topic_names, articles_table, head=articles_table_head):
    markdown_table = head

    arxiv_link = "[{0}](https://arxiv.org/pdf/{0}.pdf)" + "&nbsp;" * 4

    for j, topic in enumerate(topic_names):

        ref_s = ""
        for article_id in articles_table[j]:
            ref_s += str(arxiv_link.format(article_id))

        markdown_table += "| {0} | {1} |\n".format(topic, ref_s)

    display(Markdown(markdown_table))


def lda_articles(volume):
    titles, articles = relevant_articles(volume)
    lda_print(titles, articles)


def terms_evo(keys, se, y1, y2):
    raw_table1 = [line.strip().split(";") 
            for line in open("../stat/freq/{}.{}.csv".format(se, y1), "r").readlines()[1:]]

    raw_table2 = [line.strip().split(";") 
            for line in open("../stat/freq/{}.{}.csv".format(se, y2), "r").readlines()[1:]]

    table1 = {x[0]: float(x[1]) for x in raw_table1}
    table2 = {x[0]: float(x[1]) for x in raw_table2}

    delta = []
    for k in keys:
        delta.append([k, round(table2[k] - table1[k], 2)])

    delta = sorted(delta, key=lambda x: x[1])

    neg_labels = [x[0] for x in delta if x[1] < 0]
    neg_val = [x[1] for x in delta if x[1] < 0]

    pos_labels = [x[0] for x in delta if x[1] > 0]
    pos_val = [x[1] for x in delta if x[1] > 0]

    data= [go.Bar(
            marker = dict(color='red'),
                    x=neg_labels,
                    hoverinfo = "value",
                    y=neg_val),
           go.Bar(
            marker = dict(color='red'),
                    x=pos_labels,
                    hoverinfo = "value",
                    y=pos_val)]

    layout = go.Layout(showlegend=False, xaxis = dict(tickangle=-40), width = 600, height = 400, margin=go.Margin(t=10))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, show_link=False)


def keys_top(keys, se, y):
    raw_table = [line.strip().split(";") 
            for line in open("../stat/freq/{}.{}.csv".format(se, y), "r").readlines()[1:]]

    table = [[x[0], float(x[1])] for x in raw_table]
    table = sorted(table, key=lambda x: x[1], reverse=True)
    
    data = [go.Bar(
            marker = dict(color='green'),
                    x=[z[0] for z in table],
                    y=[z[1] for z in table])]

    layout = go.Layout(showlegend=False, xaxis = dict(tickangle=-40), width = 600,
                        height = 400, margin=go.Margin(t=10))
    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig, show_link=False)


def word_cloud(model, terms, extented_terms, n_sat = 100):
    terms_u = terms + extented_terms
    word_vecs = [model[x] for x in terms_u]
    vis_vec = TSNE().fit_transform(word_vecs)
    vis_x = [x[0] for x in vis_vec]
    vis_y = [x[1] for x in vis_vec]

    origin = {x[0]: term for term in terms for x in model.most_similar([term], topn=100)}

    clusters = [[] for x in range(len(terms))]
    for k, clust in enumerate(terms):
        for j, p in enumerate(extented_terms):
            if origin[p] == clust:
                clusters[k].append([clust, p, vis_x[j], vis_y[j]])

    far_dist = {}
    for term in terms:
        sats = model.most_similar([term], topn=n_sat) 
        far_dist[term] = int(model.similarity(term, sats[-1][0]) * 100)

    scaler = StandardScaler()
    scaler.fit(array(list(far_dist.values())).astype(float64).reshape(-1,1))

    for k in far_dist.keys():
        far_dist[k] = 1.0 - scaler.transform(far_dist[k])[0][0]

    data, annotations = [], []
    mean_ts = 17
    delta_ts = 5
    margin = 10
    annotation_template = "word: {}<br>origin: {}<br>dist: {}"
    for j, clust in enumerate(clusters):
        xc = [c[2] for c in clust]
        yc = [c[3] for c in clust]
        dists = [round(model.similarity(c[0], c[1]), 3) for c in clust]
        text = [annotation_template.format(c[1], c[0], dists[k]) for k, c in enumerate(clust)] 
        data.append(go.Scatter(
                            x = xc,
                            y = yc,
                            text = text,
                            mode = 'markers',
                            hoverinfo = "text",
                            marker = dict(
                                        size = 6.5,
                                        color = 'red',
                                        opacity = 0.5
                                    )))

        annotation_size = mean_ts + delta_ts * far_dist[terms[j]]
        annotations.append(go.Annotation(
                                         x = mean(xc), 
                                         y = mean(yc), 
                                         xanchor = "center",
                                         showarrow = False,
                                         text = "<b>{}</b>".format(terms[j]), 
                                         font = dict(size = annotation_size, 
                                                     color = 'black')))
    layout = go.Layout(
        width = 780, 
        height = 700,
        margin = go.Margin(l=margin, r=margin*3, b=margin, t=margin),
        showlegend = False, 
        hovermode = 'closest',
        xaxis = dict(showticklabels=False, showgrid=False, 
                     zeroline = False, color="lightgrey"),
        yaxis = dict(showticklabels=False, showgrid=False, 
                     zeroline = False, color="lightgrey"),
    )
    fig = dict(data=data, layout=layout)
    fig['layout']['annotations'] = annotations
    py.iplot(fig, show_link=False)


def terms_dist(model, terms):
    corr = [[model.similarity(terms[j], terms[k]) for j, _ in enumerate(terms)] for k, _ in enumerate(terms)]

    layout = go.Layout(width = 500, height = 500, xaxis = dict(tickangle=-40))

    blues_cmap = cm.get_cmap('Blues')
    blues = color_convert(blues_cmap, 255)

    data=[go.Heatmap(x=terms, 
                     y=terms,
                     z=corr, 
                     showscale = False,
                     colorscale = blues)]

    py.iplot({'data': data, 'layout': layout}, show_link=False)


def closest_keys(terms, model, topn=100):
    extented_terms = []
    for term in terms:
        if term:
            for x in model.most_similar([term], topn=topn):
                extented_terms.append(x[0])
        else:
            warn("There is no association for term: {}".format(term))

    return extented_terms


def extract_subsections(section, year):
    ann_path = "../arxiv/annotations"
    ann_files = glob("{}/{}.{}/*.txt".format(ann_path, section, str(year)))
    
    ann_content  = []
    for file in ann_files:
        ann_content.append("".join(open(file, "r").readlines()))
    
    articles_categories = []
    for _, ann in enumerate(ann_content):
        soup = BS(ann, "lxml")
        a_categories = soup.find_all({"category" : "term"})
        articles_categories.append(a_categories[0]["term"])
        
    return articles_categories

                               
def subsections_ratio(section, year, colormap_name="Wistia", min_percent=0):
    with open('abbreviations/{}.json'.format(section)) as handle:
        description = json.loads(handle.read())

    articles_sections = extract_subsections(section, year)
    sections_cn = Counter(articles_sections)

    sections = {re.sub("{}.".format(section), '', x) : sections_cn[x] for x in sections_cn if x.find(section) == 0}

    min_cn = sum(list(sections.values())) * min_percent * 0.01
    chart_labels, chart_values = [], []
    other_percent, other_cn = 0, 0

    for x in sorted(sections.items(), key=operator.itemgetter(1), reverse=True):
        if x[1] > min_cn:
            chart_labels.append(x[0])
            chart_values.append(x[1])
        else:
            other_percent += x[1]
            other_cn += 1

    if other_cn > 0:
        chart_labels.append("Other ({})".format(other_cn))
        chart_values.append(other_percent)

    norm_val = (np.array(chart_values) - np.min(chart_values)) / (np.max(chart_values) - np.min(chart_values))
            
    colormap = matplotlib.cm.get_cmap(colormap_name)
    pieces_colors = list(map(lambda x: matplotlib.colors.rgb2hex(colormap(x)), norm_val))

    hover_text = [description[x] for x in chart_labels if x in description]

    if other_cn > 0:
        hover_text += ["Other subsections".format(other_cn)]

    trace = [go.Pie(text=hover_text, hoverinfo="text+value",
                    textinfo= 'label+percent', labels=chart_labels,
                    textposition = 'outside', values=chart_values,
                    pull=.05, marker=dict(colors=pieces_colors))]

    layout = go.Layout(height = 450, margin=go.Margin(t=40))

    py.iplot({'data': trace, 'layout': layout}, show_link=False)


#####################


def load_articles_headers(section, year, ap="../arxiv/annotations"):
    an_files = glob("{}/{}.{}/*.txt".format(ap, section, str(year)))

    an_content = []
    for file in an_files:
        an_content.append("".join(open(file, "r").readlines()))

    headers = []
    for ann in an_content:
        soup = BS(ann, "lxml")
        a_categories = soup.find_all({"category": "term"})
        all_authors = soup.find_all('author')

        headers.append([[author.find('name').text for author in all_authors],
                        [x["term"] for x in a_categories]])
    return headers


def sort_dict(d, n=None, reverse=True):
    if not n:
        n = len(d)
    return sorted(d.items(), key=operator.itemgetter(1), reverse=reverse)[:n]


def categories_top_authors(an_headers, categories, ntop=4, max_ncat=10):
    authors_cn = {}
    for category in categories:
        cn = Counter()
        for ah in an_headers:
            if category in ah[1]:
                for author in ah[0]:
                    cn[author] += 1
        authors_cn[category] = cn

    if len(categories) > max_ncat:
        subcategories_articles_cn = {key : sum(list(authors_cn[key].values()))
                                     for key in authors_cn.keys()}
        categories = [x[0] for x in sort_dict(subcategories_articles_cn, max_ncat)]

    return {c: sort_dict(authors_cn[c], ntop) for c in categories}


def plot_top_authors(section, data, ntop=4):
    sub_sections = sorted(data.keys())

    if section + ".other" in sub_sections:
        sub_sections.remove(section + ".other")

    plot_titles = [re.sub("{}.".format(section), '', x) for x in sub_sections]

    ncol = 2
    nrow = math.ceil(len(sub_sections) / ncol)

    subplots, annotations = [], []
    for j, w in enumerate(sub_sections):
        values = data[w]

        n_articles = [x[1] for x in values[:ntop]]
        top_authors = [x[0] for x in values[:ntop]]

        subplots.append(go.Bar(
                            x=n_articles,
                            y=top_authors,
                            hoverinfo="text",
                            orientation='h',
                            text=n_articles,
                            marker=dict(color='blue')))

        for x, y in zip(n_articles, top_authors):
            annotations.append(go.Annotation(
                                            x=x, y=y, text=x,
                                            xanchor="left",
                                            xref="x{}".format(j + 1),
                                            yref="y{}".format(j + 1),
                                            showarrow=False,
                                            font=dict(size=11, color='lightgrey')))

    fig = tools.make_subplots(rows=nrow, cols=ncol,
                              subplot_titles=plot_titles,
                              horizontal_spacing=0.35,
                              vertical_spacing=0.09,
                              print_grid=False)
    for i in range(nrow):
        for j in range(ncol):
            idx = i * 2 + j
            if idx > len(sub_sections) - 1:
                break

            fig.append_trace(subplots[idx], i + 1, j + 1)
            fig['layout']['yaxis{0}'.format(idx + 1)].update(ticklen=3, autorange="reversed")
            fig['layout']['xaxis{0}'.format(idx + 1)].update(showticklabels=False, showgrid=False)
            fig['layout']['yaxis{0}'.format(idx + 1)]['tickfont'].update(size=11, color="lightgrey")

    fig['layout'].update(height=150 * nrow, width=350 * ncol,
                         showlegend=False, title="",
                         margin=go.Margin(t=60, l=140, b=30))

    fig['layout']['annotations'].extend(annotations)

    py.iplot(fig, show_link=False)


def active_authors(section, year):
    headers = load_articles_headers(section, year)

    articles_categories = [x[1] for x in headers]
    raw_categories = set(reduce(lambda x, y: x + y, articles_categories))
    categories = [x for x in raw_categories if x.find(section) == 0]

    authors = categories_top_authors(headers, categories)
    plot_top_authors(section, authors)


##############


def basename(fn):
    return fn.split("/")[-1]


def group_articles_by_subsection(section, year, n_files=1000):
    annotations_fn = glob("{}/{}.{}/*.txt".format(annotations_path, section, str(year)))

    annotations_content = []
    for file in annotations_fn[:n_files]:
        annotations_content.append("".join(open(file, "r").readlines()))

    sub_content = {}
    for j, ann in enumerate(annotations_content[:n_files]):
        soup = BS(ann, "lxml")
        article_categories = soup.find_all({"category": "term"})

        article = basename(annotations_fn[j])

        for category in [x["term"] for x in article_categories]:
            if section in category:
                if category in sub_content:
                    sub_content[category].append(article)
                else:
                    sub_content[category] = [article]
    return sub_content


def get_article_month(s):
    return s[2:4]


def load_subsections_content(section, year, n_articles = 1000):
    texts_path = "../arxiv/{0}/{1}/".format(section, year)

    groups = group_articles_by_subsection(section, year)

    all_articles = []
    for z in range(12):
        all_articles += glob("{}{}/*.txt".format(texts_path, str(z + 1).zfill(2)))

    all_texts = shared.load_texts(all_articles, "{}.{}".format(section, year), True)

    sections_content = []
    for articles_group in groups.values():

        fns = list(map(lambda x: "{}{}/".format(texts_path, get_article_month(x)) + x,
                                                articles_group))
        fns = [x for x in fns if isfile(x)]

        sentences = [all_texts[x] for x in fns]

        sections_content.append(" ".join(reduce(lambda x, y: x + y, sentences)))

    return groups, sections_content


def count_terms(sections_content, terms):
    cn_matrix = []
    for t in terms:
        cn_matrix.append([content.count(t) for content in sections_content])
    return cn_matrix


def terms_subsections_plot(data_matrix, terms, subsections,
                           pw=11, ph=5, fs=12, cm='Greens'):

    plt.figure(figsize=(pw, ph))

    plt.xticks(range(len(terms)), terms, rotation=90, size=fs)
    plt.yticks(range(len(subsections)), subsections, size=fs)

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(terms), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subsections), 1), minor=True)

    plt.grid(which="minor", color="w", linestyle='-', linewidth=1.0)

    plt.imshow(np.array(data_matrix).T, cmap=cm)


def terms_subsections_occur(section, year, terms):
    groups, content = load_subsections_content(section, year)
    data_matrix = count_terms(content, terms)
    subsections = [re.sub("{}.".format(section), '', x) for x in groups.keys()]
    terms_subsections_plot(data_matrix, terms, subsections)
