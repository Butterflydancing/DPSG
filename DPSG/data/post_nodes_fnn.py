#!/usr/bin/env python
# coding: utf-8


import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing
from multiprocessing import Pool, Manager


def create_FNN_post_nodes(dataset, folder_name):
    
    data_path = 'processed_data/FakeNewsNet/%s/' % dataset # fake news net page
    combo_path = 'processed_data/FakeNewsNet/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data
    image_path = 'processed_data/FakeNewsNet/%s/visual_features/' % dataset
    post_path = 'processed_data/FakeNewsNet/%s/%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here
    user_path = 'processed_data/FakeNewsNet/%s/%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = 'processed_data/FakeNewsNet/%s/%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here
    roberta_path = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/%s/text_embeddings/tweet_text/'%dataset # get user roberta embedding here

    neighbors_path = 'rwr_results/%s/' %folder_name # read neighbor list from here


    if not os.path.exists(post_path):
        os.makedirs(post_path)

    post_id = []
    with open(neighbors_path + 'n_neighbors.txt', 'r') as f:
        news_neighbors = f.readlines()
    for news in tqdm(news_neighbors, desc='get all neighbors...'):
        news = news.split()
        for neighbor in news[1:]:
            if neighbor[0] == 'p' and neighbor[1:] != 'PADDING':
                post_id.append((neighbor[1:]).split('t')[0])

    print('get all post features...')
    f = open(data_path + 'post_features_onehot.txt', 'r')
    post_f = f.readlines()
    f.close()
    post_feature = dict()
    for line in tqdm(post_f, desc = 'get all post features'):
        line = line.split(' ', 1)
        post_feature[line[0][1:-1]] = line[1]

    int_features = list()
    for i in post_feature.keys():
        int_features.append(list(map(float, post_feature[i].split())))
    padding_features = np.mean(int_features, 0)
    padding_features = list(map(str, padding_features))


    if os.path.exists('post_content.txt'):
        print("load post content from txt file..")
        with open('post_content.txt', 'r') as f:
            post_content = np.loadtxt(f).astype(float)
    else:
        post_content = dict()

        for post in tqdm(post_id, desc='get all post content'):
            try:
                f = open(roberta_path + post + '.txt', 'r')
                description = np.loadtxt(f, delimiter = ' ')
                f.close()
                post_content[post] = description
            except:
                pass


    print('normalize post content...')
    scaler = preprocessing.StandardScaler().fit(list(post_content.values()))
    normalized_content = scaler.transform(list(post_content.values()))
    keys = list(post_content.keys())
    for i in range(len(keys)):
        post_content[keys[i]] = list(map(str, normalized_content[i]))

    padding_description = ['0'] * len(normalized_content[0])

    for batch in tqdm(range(len(post_id)//5000 + 1), desc='writing batches......'):
        with open(post_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing post nodes.....'):
                if (i >= len(post_id)):
                    break
                f.write('p ' + post_id[i] + '\n')
                try:
                    f.write(post_feature[post_id[i]])
                except:
                    f.write(' '.join(padding_features) + '\n')
                try:
                    f.write(' '.join(post_content[post_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_description) + '\n')



