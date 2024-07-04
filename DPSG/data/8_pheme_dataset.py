
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing

from multiprocessing import Pool
from post_nodes_pheme import create_post_nodes
from user_nodes_pheme import create_user_nodes
from news_nodes_pheme import create_news_nodes

def create_pheme_dataset(node_type, folder_name):
    
    if node_type == 'news':
        create_news_nodes(folder_name)
    elif node_type == 'post':
        create_post_nodes(folder_name)
    elif node_type == 'user':
        create_user_nodes(folder_name)
        
        
if __name__ == '__main__':
    
    num_process = 3
    with Pool(num_process) as p:
        p.starmap(create_pheme_dataset, [('news', 'pheme_n5_p5_u100')])

