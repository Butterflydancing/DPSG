#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing

from multiprocessing import Pool, Manager

def create_FNN_user_nodes(dataset, folder_name):

    data_path = 'processed_data/FakeNewsNet/%s/' % dataset # fake news net page
    combo_path = 'processed_data/FakeNewsNet/%s/%s/' % (dataset, folder_name) # store 5n5p100u input data
    roberta_path = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/%s/text_embeddings/user_description/'% dataset # get user roberta embedding here
    image_path = 'processed_data/FakeNewsNet/%s/visual_features/' % dataset
    post_path = 'processed_data/FakeNewsNet/%s/%s/normalized_post_nodes/' % (dataset, folder_name) # store post nodes here
    user_path = 'processed_data/FakeNewsNet/%s/%s/normalized_user_nodes/' % (dataset, folder_name) # store user nodes here
    news_path = 'processed_data/FakeNewsNet/%s/%s/normalized_news_nodes/' % (dataset, folder_name) # store news nodes here

    neighbors_path = 'rwr_results/%s/' %folder_name # read neighbor list from here


    if not os.path.exists(user_path):
        os.makedirs(user_path)

    user_id = []
    with open(neighbors_path + 'n_neighbors.txt', 'r') as f:
        news_neighbors = f.readlines()
    for news in tqdm(news_neighbors, desc='get all neighbors...'):
        news = news.split()
        for neighbor in news[1:]:
            if neighbor[0] == 'u' and neighbor[1:] != 'PADDING':
                user_id.append((neighbor[1:]).split('t')[0])

    # user features
    print('get all user features...')
    f = open(data_path + 'user_features_onehot.txt', 'r')
    
    user_f = f.readlines()
    f.close()
    user_feature = dict()
    for line in tqdm(user_f, desc = 'get all user features'):
        line = line.split(' ', 1)
        user_feature[line[0][1:-1]] = line[1]

    # padding features
    int_features = list()
    for i in user_feature.keys():
        int_features.append(list(map(float, user_feature[i].split())))
    padding_features = np.mean(int_features, 0)
    padding_features = list(map(str, padding_features))

    # user description
    if os.path.exists('user_description.txt'):
        print("load user description from txt file..")
        with open('user_description.txt', 'r') as f:
            user_d = np.loadtxt(f).astype(float)
    else:
        user_d = dict()
    
        for user in tqdm(user_id, desc='get all user description'):
            try:
                f = open(roberta_path + user + '.txt', 'r')
                description = np.loadtxt(f, delimiter = ' ')
                f.close()
                if len(description) == 0:
                    continue
                user_d[user] = description
            except:
                pass

    print('normalize user description...')
    scaler = preprocessing.StandardScaler().fit(list(user_d.values()))
    normalized_d = scaler.transform(list(user_d.values()))
    keys = list(user_d.keys())
    for i in range(len(keys)):
        user_d[keys[i]] = list(map(str, normalized_d[i]))

    padding_d = ['0'] * len(normalized_d[0])

    for batch in tqdm(range(len(user_id)//5000 + 1), desc='writing batches......'):
        with open(user_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing user nodes.....'):
                if (i >= len(user_id)):
                    break
                f.write('u ' + user_id[i] + '\n')
                try:
                    f.write(user_feature[user_id[i]].strip('\n') + '\n')
                except:
                    f.write(' '.join(padding_features) + '\n')
                try:
                    f.write(' '.join(user_d[user_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_d) + '\n')





