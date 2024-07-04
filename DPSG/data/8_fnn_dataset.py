#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing
from multiprocessing import Pool
from post_nodes_fnn import create_FNN_post_nodes
from user_nodes_fnn import create_FNN_user_nodes
from news_nodes_fnn import create_FNN_news_nodes

def create_FNN_dataset(node_type, dataset, folder_name):
    if node_type == 'news':
        create_FNN_news_nodes(dataset, folder_name)
    elif node_type == 'post':
        create_FNN_post_nodes(dataset, folder_name)
    elif node_type == 'user':
        create_FNN_user_nodes(dataset, folder_name)
   

if __name__ == '__main__':
    
    num_process = 3
    with Pool(num_process) as p:
        p.starmap(create_FNN_dataset, [
                                        ('news', 'gossipcop', 'fnn_gossipcop_50'),
                                       ('post', 'gossipcop', 'fnn_gossipcop_50'),
                                       ('user', 'gossipcop', 'fnn_gossipcop_50'),
                                       # ('news', 'politifact', 'fnn_politifact_n5_p5_u100'),
                                       # ('post', 'politifact', 'fnn_politifact_n5_p5_u100'),
                                       # ('user', 'politifact', 'fnn_politifact_n5_p5_u100'),
                                       ])
