from networkx.drawing.nx_agraph import graphviz_layout
from IPython.display import display, Markdown
from openpyxl import load_workbook
from gensim.models import Word2Vec
from matplotlib import cm
import networkx as nx
from numpy import *

import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools

from warnings import warn
import csv


chart_count = 0
cite_table = \
"""
| {0}Author(s){1} | {0}Year{1} | {0}Ref. count{1} |
| --- | --- | --- |
""".format("<center>", "<center>")


def replics(model, target_word, topn=10):
    return {word: model.wv.vocab[word].count
                           for word in model.wv.vocab.keys()
                           if word.find(target_word) == 0}


def topics_normalize(model, raw_topics):
    norm_topics = []
    for word in raw_topics:
        r = replics(model, word)
        if r is not None:
            norm_topics.append(max(r, key=r.get))
        else:
            norm_topics.append(None)

    return norm_topics


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_lines(fn):
    return [line.strip() for line in open(fn, "r").readlines()]


def color_convert(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(uint8, array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale


def draw_heatmap(volume, terms):
    corr_path = "../stat/frequency/{}/terms_similar.csv".format(volume)
    corr = []
    with open(corr_path, "rt") as f:
        reader = csv.reader(f)
        next(f)
        next(f)
        for line in reader:
            corr.append([float(x) for x in line[1:]])

    blues_cmap = cm.get_cmap('Blues')
    blues = color_convert(blues_cmap, 255)

    layout = go.Layout(
        xaxis = dict(
            ticktext=terms,
            tickvals=[i for i in range(len(corr))],
            tickangle=-45,
            linewidth=1,
            mirror=True
        ),
        yaxis = dict(
            ticktext=terms,
            tickvals=[i for i in range(len(corr))],
            tickangle=-45,
            linewidth=1,
            mirror=True
        ),
        width = 500,
        height = 500
        )

    trace = go.Heatmap(z = corr,
                       colorscale = blues,
                       showscale = False)
    data=[trace]
    py.iplot({'data': data, 'layout': layout}, show_link=False)


def add_edges(G, unique_pairs, terms, model, min_dist):
    w = []
    for pair in unique_pairs:
        word_one = terms[pair[0]]
        word_two = terms[pair[1]]
        dist = model.similarity(word_one, word_two)
        if dist > min_dist:
            w.append(edge_width)
        else:
            w.append(0)
        G.add_edge(word_one, word_two)
    return w

def draw_graph(terms, extented_terms, model):
    draw_method = "neato"
    min_dist1 = 0.3
    min_dist2 = 0.85
    font_small = 10
    font_large = 15

    terms_u = terms + extented_terms

    G = nx.Graph()
    G.add_nodes_from(terms_u)
    p = graphviz_layout(G, prog=draw_method)

    trace = go.Scatter(
        x = [p[word][0] for word in terms_u],
        y = [p[word][1] for word in terms_u],
        mode = 'text',
        name = 'None',
        text = ["<b>" + x + "</b>" if x in terms else x for x in terms_u],
        textposition = 'center',
        textfont = {
                    "size": [font_large if x in terms else font_small for x in terms_u]
                }
    )
    shapes = []
    unique_pairs = [[i, j] for j in range(len(terms_u)) for i in range(j)]

    for pair in unique_pairs:
            word_one = terms_u[pair[0]]
            word_two = terms_u[pair[1]]
            dist = model.similarity(word_one, word_two)
            if word_one in terms and word_two in terms:
                if dist > min_dist1:
                    shapes.append({
                        'type': 'line',
                        'x0': p[word_one][0],
                        'y0': p[word_one][1],
                        'x1': p[word_two][0],
                        'y1': p[word_two][1],
                        'line': {
                                'color': "red",
                                'width': 1.0
                                },
                        'layer' : 'below',
                        'opacity' : 0.65
                        })
            else:
                if dist > min_dist2:
                    shapes.append({
                        'type': 'line',
                        'x0': p[word_one][0],
                        'y0': p[word_one][1],
                        'x1': p[word_two][0],
                        'y1': p[word_two][1],
                        'line': {
                                'color': "lightskyblue",
                                'width': 1.0
                                },
                        'layer' : 'below',
                        'opacity' : 0.65
                        })

    layout = go.Layout(
        showlegend=False,
        shapes = shapes,
        xaxis=dict(
            showgrid=False,
            showticklabels = False,
            zeroline=False),
        yaxis=dict(
            showgrid=False,
            showticklabels = False,
            zeroline=False),
        width=700,
        height=700,
        hovermode= 'closest'
    )

    data = [trace]
    py.iplot({'data': data, 'layout': layout}, show_link=False)


def draw_scatter(volume):
    n_words = 50

    wb = load_workbook(filename = '../stat/frequency/{}/topics.xlsx'.format(volume))

    sheet_list = wb.get_sheet_names()
    sheet_list.remove('uncovered')

    data = []
    for name in sheet_list:
        ws = wb[name]

        if min(ws.max_row, ws.max_column) <= 1:
            warn("Empty sheet '{}'".format(name))
            continue

        table = [[line[0].value, line[1].value, line[2].value] for line in
                    zip(list(ws.columns)[0], list(ws.columns)[1], list(ws.columns)[2])]

        labels = [line[0] for line in table]
        x = [line[1] for line in table]
        y = [line[2] for line in table]

        data.append(dict(
                        type = 'scatter',
                        x = [line[1] for line in table][:n_words],
                        y = [line[2] for line in table][:n_words],
                        mode = 'markers',
                        hoverinfo = "text",
                        hoveron = "points",
                        name = name,
                        text = [line[0] for line in table][:n_words]
                    ))

    layout = dict(
            hovermode= 'closest',
            xaxis=dict(title="tf-idf"),
            yaxis=dict(title="frequency")
            )

    py.iplot({'data': data, 'layout': layout}, validate=False, show_link=False)


def add_bar(fig, inp, n_topics, n_keys, n_row, n_col):
    global chart_count

    if chart_count >= n_topics:
        return

    labels = sorted([x[1] for x in inp])
    values = sorted([x[0] for x in inp])

    data = go.Bar(
            x=labels[:n_keys],
            y=values[:n_keys],
            orientation = 'h',
            name = "topic #{}".format(chart_count + 1),
            marker = dict(color='orange')
    )
    fig.append_trace(data, (chart_count // n_col) + 1, (chart_count % n_col) + 1)
    fig['layout']['xaxis{0}'.format(chart_count+1)].update(showticklabels = False)
    fig['layout']['yaxis{0}'.format(chart_count+1)].update(ticklen=3)
    fig['layout']['yaxis{0}'.format(chart_count+1)]['tickfont'].update(size=11,
                                                                       color="lightgrey")
    chart_count += 1

def draw_barchart(volume, n_keys = 10, n_row = 5, n_col = 3):
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

    for topic_words in table[:n_charts]:
        add_bar(fig, topic_words, n_topics, n_keys, n_row, n_col)

    fig['layout'].update(height=750, width=750, showlegend=False)
    py.iplot(fig, show_link=False);


def cites(volume):
    global cite_table
    n_cites = 10

    with open("../stat/references/{}.txt".format(volume), "r") as f:
        data = f.readlines()

    for line in data[:n_cites]:
        authors, year, count = line.split(",")

        cite_table += "|{0}{2}{1}|{0}{3}{1}|{0}{4}{1}|\n".format("<center>",
                                                            "</center>",
                                                           ", ".join(map(lambda x: x.strip(), authors.split("&"))),
                                                            year.strip(),
                                                            count.strip())

    display(Markdown(cite_table))

def top_chart(volume):
    data = open("../stat/frequency/{0}/terms_unique.csv".format(volume), "r").readlines()[2:]
    keywords = []
    for line in data:
        word, val, _ = line.split(",")
        keywords.append([word, float(val)])

    from pprint import pprint
    keywords_s = sorted(keywords, key = lambda x: x[1], reverse=True)

    data = [go.Bar(
                x=[x[0] for x in keywords_s],
                y=[x[1] for x in keywords_s]
    )]
    py.iplot(data, show_link = False)
