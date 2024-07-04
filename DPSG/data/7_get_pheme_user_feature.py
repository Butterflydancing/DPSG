#!/usr/bin/env python
# coding: utf-8


import os
from tqdm import tqdm
import json

data_path = 'PHEME/'
output_dir = 'processed_data/PHEME/'

event_list = ['charliehebdo-all-rnr-threads/','ferguson-all-rnr-threads/',
          'germanwings-crash-all-rnr-threads/', 'ottawashooting-all-rnr-threads/', 'sydneysiege-all-rnr-threads/']

user_features = list()

for event in tqdm(event_list):
    for news_type in ['rumours/', 'non-rumours/']:

        event_path = data_path + event + news_type
        post_list = os.listdir(event_path)
        try:
            post_list.remove('.DS_Store')
        except:
            pass
        for post in tqdm(post_list):
            for post_type in ['/source-tweet/', '/reactions/']:
                tweet_path = event_path + post + post_type
                tweet_list = os.listdir(tweet_path)
                try:
                    tweet_list.remove('.DS_Store')
                except:
                    pass
                for tweet in tweet_list:
                    with open(tweet_path + tweet) as json_file:
                        tweet_dict = json.load(json_file)
                    user_info = tweet_dict['user']
                    user_id = user_info['id_str']
                    verified = str(user_info['verified'] - 0)#True:1,False:0
                    statuses_count = str(user_info['statuses_count'])
                    followers_count = str(user_info['followers_count'])
                    friends_count = str(user_info['friends_count'])
                    geo_enabled = str(user_info['geo_enabled'] - 0)
                    favorites_count = str(user_info['favourites_count'])
                    
                    user_features.append([user_id, verified, statuses_count, followers_count,
                                          favorites_count, friends_count, geo_enabled])

from sklearn import preprocessing
import numpy as np

user_features=np.array(user_features)
matrix=user_features[:,[2,3,4,5]]
nor=preprocessing.MinMaxScaler()
nor_matrix=nor.fit_transform(matrix)

user_features=user_features.tolist()
nor_matrix=nor_matrix.tolist()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
with open(output_dir + 'onehot_user_features_pheme.txt', 'w') as f:
    for i,j in zip(user_features,nor_matrix):
        f.write('%s: %s %s %s %s %s %s\n' %(i[0], i[1], j[0], j[1],
                                              j[2], j[3], i[-1],))
    f.write('%s: %s %s %s %s %s %s\n' % ('uPADDING', '0', '0', '0',
                                         '0', '0', '0',))
