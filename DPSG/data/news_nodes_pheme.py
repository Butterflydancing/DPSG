#!/usr/bin/env python
# coding: utf-8

import numpy as np
import json
import os
from tqdm import tqdm
from sklearn import preprocessing


def create_news_nodes(folder_name):
    
    news_text_path = 'PHEME/text_embeddings/tweet_text/' # get user roberta embedding here
    user_text_path = 'PHEME/text_embeddings/user_description/'

    data_path = 'processed_data/PHEME/' # fake news net page
    combo_path = 'processed_data/PHEME/%s/' % folder_name # store 5n5p100u input data
    post_path = 'processed_data/PHEME/%s/normalized_post_nodes/' % folder_name # store post nodes here
    user_path = 'processed_data/PHEME/%s/normalized_user_nodes/' % folder_name # store user nodes here
    news_path = 'processed_data/PHEME/%s/normalized_news_nodes/' % folder_name# store news nodes here

    neighbors_path = 'rwr_results/%s/' % folder_name # read neighbor list from here
#-------------------------------news-----------------------------------------------------
    if not os.path.exists(news_path):
        os.makedirs(news_path)

    print('get news labels...')
    with open (data_path + 'news_label.txt', 'r') as f:
        data = f.readlines()
    news_label = dict()
    for line in data:
        line = line.rstrip('\n').split(': ')
        news_label[line[0]] = line[1] 

    print("load news neighbors.txt")
    with open(neighbors_path + 'n_neighbors.txt', 'r') as f:
        news_neighbors = f.readlines()

    print('get all neighbors...')
    all_post_neighbors = []
    all_user_neighbors = []
    all_news_neighbors = []
    news_id = []
    post_id=[]
    user_id=[]
    for news in tqdm(news_neighbors, desc='get all neighbors...'):
        news = news.split()
        news_id.append((news[0][1:-1]).split('t')[0])

        n_neighbors = []
        p_neighbors = []
        u_neighbors = []
        for neighbor in news[1:]:
            if neighbor[0] == 'p':
                p_neighbors.append(neighbor[1:])
                if neighbor[1:].split('t')[0] !='PADDING':
                    post_id.append(neighbor[1:].split('t')[0])
            elif neighbor[0] == 'u':
                u_neighbors.append(neighbor[1:])
                if neighbor[1:].split('t')[0] !='PADDING':
                    user_id.append(neighbor[1:].split('t')[0])
            elif neighbor[0] == 'n':
                n_neighbors.append(neighbor[1:])
        all_post_neighbors.append(p_neighbors)
        all_user_neighbors.append(u_neighbors)
        all_news_neighbors.append(n_neighbors)

    news_content = dict()
    for news in tqdm(news_id, desc='get all news content'):

        try:
            f = open(news_text_path + news + '.txt', 'r')
            content = np.loadtxt(f, delimiter = ' ')
            f.close()
            news_content[news] = content
        except:
            pass

    print('normalize news content...')   
    scaler = preprocessing.StandardScaler().fit(list(news_content.values()))
    normalized_content = scaler.transform(list(news_content.values()))
    keys = list(news_content.keys())
    for i in range(len(keys)):
        news_content[keys[i]] = list(map(str, normalized_content[i]))

    padding_content = ['0'] * len(normalized_content[0])
    for batch in tqdm(range(len(news_id)//5000 + 1), desc='writing batches......'):
        with open(news_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing news nodes.....'):
                if (i >= len(news_id)):
                    break
                f.write('n ' + news_id[i] + ' %s' %news_label[news_id[i]] + '\n')
                try:
                    f.write(' '.join(news_content[news_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_content) + '\n')
                f.write(' '.join(all_news_neighbors[i]) + '\n')
                f.write(' '.join(all_post_neighbors[i]) + '\n')
                f.write(' '.join(all_user_neighbors[i]) + '\n')

    #-------------------post---------------------------
    if not os.path.exists(post_path):
        print('create directory..')
        os.makedirs(post_path)

    print("load post neighbors.txt")
    post_content = dict()
    for post in tqdm(post_id, desc='get all post content'):
        try:
            f = open(news_text_path + post + '.txt', 'r')
            description = np.loadtxt(f, delimiter=' ')
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

    # padding description
    padding_content = ['0'] * len(normalized_content[0])
    for batch in tqdm(range(len(post_id)//5000 + 1), desc='writing batches......'):
        with open( post_path + 'batch_%d.txt' %batch, 'w') as f:
            for i in tqdm(range(batch*5000, (batch+1)*5000), desc='writing post nodes.....'):
                if (i >= len(post_id)):
                    break
                f.write('p ' + post_id[i] + '\n')
                try:
                    f.write(' '.join(post_content[post_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_content) + '\n')

    # -------------------user---------------------------
    if not os.path.exists(user_path):
        print('create directory')
        os.makedirs(user_path)

    # user features
    print('get all user features...')
    f = open(data_path + 'onehot_user_features_pheme.txt', 'r')
    user_f = f.readlines()
    f.close()
    user_feature = dict()
    for line in tqdm(user_f, desc = 'get all user features'):
        line = line.split(': ')
        user_feature[line[0]] = line[1]


    user_d = dict()
    for user in tqdm(user_id, desc='get all user description'):
        try:
            f = open(user_text_path + user + '.txt', 'r')
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
                    f.write(user_feature[u_neighbors[i]].strip('\n') + '\n')
                except:
                    f.write(user_feature['uPADDING'].strip('\n') + '\n')
                try:
                    f.write(' '.join(user_d[user_id[i]]) + '\n')
                except:
                    f.write(' '.join(padding_d) + '\n')


