"""
pheme{
    "n": 6425,5802
    "p": 98929,97410
    "u": 51043,49778}
politifact{
    "n": 870,884,733
    "p": 440467,1317371,1167624,
    "u": 468210,686925,627191}

gossipcop{
    "n": 21068,21202,20955,
    "p": 1192766,1920964,1890915,
    "u": 429628,630611,618440}

"""
import os
import json


def stats(in_dir, node_types, edge_files):
    nodes = {t : [] for t in node_types}
    for (t0, t1), fname in edge_files.items():
        with open(os.path.join(in_dir, fname), 'r') as f:
            for line in f.readlines():
                info = line.strip().split()
                nodes[t0].append(info[0])
                nodes[t1].append(info[1])
    counts = {k : len(set(v)) for k, v in nodes.items()}
    return counts

if __name__ == "__main__":
    for dataset in ['politifact', 'gossipcop','pheme']:
        if dataset in ['politifact', 'gossipcop']:
            prefix = f'fnn_{dataset}_'
            in_dir = f'FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def/{dataset}'
            edge_dir = os.path.join(in_dir, dataset)
            node_types = ['n', 'p', 'u']
            edge_files = {
                ('n', 'n'): 'news-news edges.txt',
                ('n', 'p'): 'news-post edges.txt',
                ('p', 'u'): 'post-user edges.txt',
                ('u', 'u'): 'user-user edges.txt',
            }
            edges_to_enforce = {('p', 'u'),}
        elif dataset == 'pheme':
            prefix = 'pheme_'
            in_dir = 'PHEME'
            edge_dir = in_dir
            node_types = ['n', 'p', 'u']
            edge_files = {
                ('n', 'p'): 'PhemeNewsPost.txt',
                ('n', 'u'): 'PhemeNewsUser.txt',
                ('p', 'p'): 'PhemePostPost.txt',
                ('p', 'u'): 'PhemePostUser.txt',
                ('u', 'u'): 'PhemeUserUser.txt',
            }
            edges_to_enforce = {('n', 'u'), ('p', 'u'),}

        counts = stats(in_dir, node_types, edge_files)
        print(dataset)
        print(json.dumps(counts, indent=4))