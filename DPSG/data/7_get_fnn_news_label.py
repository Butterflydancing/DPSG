#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm


def get_news_label(dataset):
    
    news_label = list()
    print('reading %s...' %dataset)
    pathway = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/%s/' % dataset
    output_dir = 'processed_data/FakeNewsNet/%s/' % dataset
    
    for news_type in ['fake/', 'real/']:
        print('reading news info from %s file...' %news_type)
        if news_type == 'fake/':
            label = '0'
        else:
            label = '1'
        news_list = os.listdir(pathway + news_type)
        for n in news_list:
            n_id=n[len("politifact"):] if n[:len("politifact")] == "politifact" else n[len("gossipcop-"):]
            news_label.append([n_id, label])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_dir + 'news_label.txt', 'w') as f:
        for news in news_label:
            f.write(' '.join(news) + '\n')

if __name__ == '__main__':
    
    get_news_label('politifact')
    get_news_label('gossipcop')
