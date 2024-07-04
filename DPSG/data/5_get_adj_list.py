
import os
import json
from tqdm import tqdm

def add_adjacent(m, n):
    if m not in adj_list.keys():
        adj_list[m] = []
    adj_list[m].append(n)


in_dir = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def'
# edge_dir = os.path.join(in_dir, 'politifact')
edge_dir = os.path.join(in_dir, 'gossipcop')

edge_files = {
    ('n', 'n'): 'news-news edges.txt',
    ('n', 'p'): 'news-post edges.txt',
    ('p', 'u'): 'post-user edges.txt',
    ('u', 'u'): 'user-user edges.txt',
}
output_dir = f"rwr_results/fnn_gossipcop_n5_p5_u100"

adj_list = dict()

for (main_type, neig_type), edge_f in edge_files.items():
    with open(os.path.join(edge_dir, edge_f), "r") as f:
        for l in tqdm(f.readlines(), desc='read ' + main_type+' '+neig_type):
            l = l.strip().split()
            add_adjacent(main_type + l[0], neig_type + l[1])
            add_adjacent(neig_type + l[1], main_type + l[0])


with open(output_dir+"/original_adj",'w') as f:
    print(len(adj_list))
    json.dump(adj_list,f)