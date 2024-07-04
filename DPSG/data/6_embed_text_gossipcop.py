import os
import json
import torch
from tqdm import tqdm
from text_embedder import TextEmbedder
from multiprocessing import Manager, Process, Pool

in_dir = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset'
dataset = 'gossipcop'
ds_dirs = [os.path.join(in_dir, dataset, ss) for ss in ['real', 'fake']]
involved_dir = f'rwr_results/fnn_{dataset}_u50/'

out_dir = f'FakeNewsNet-Dataset/FakeNewsNet_Dataset/{dataset}/text_embeddings'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

configs = {
    'news' : {
        'model name' : 't5-base-finetuned-summarize-news',
        'batch size' : 4,
    },
    'tweet' : {
        'model name' : 'bertweet-base',
        'batch size' : 256,
    },
}
device = 'cpu'

def embed_text_worker(texts, max_seq_len, config, rank, return_dict):
    batch_size = config['batch size']
    num_batches = (len(texts) + batch_size - 1) // batch_size
    embedder = TextEmbedder(max_seq_len, config['model name'], device=device)
    features = torch.zeros(len(texts), embedder.embed_dim)
    for i in tqdm(range(num_batches), desc='embed text'):
        mn = i * batch_size
        mx = min(len(texts), (i + 1) * batch_size)
        features[mn:mx] = embedder(texts[mn:mx])[:, 0, :].squeeze(1)
    return_dict[rank] = features
    del embedder

def save_embed_file(dir, tid, feature):
    f = open('{}/{}.txt'.format(dir, tid), 'w')
    f.writelines(' '.join(['{:.8f}'.format(v) for v in feature]) + '\n')
    f.close()

def save_embeddings_worker(dir_name, ids, features):
    for i in tqdm(range(len(ids)), desc='save embed'):
        save_embed_file(os.path.join(out_dir, dir_name), ids[i], features[i])

def embed_text(ids, texts, max_seq_len, config, dir_name, num_process = 8):
    def skip_existed(id_list, text_list, dir_name):
        print('skip existed...')
        new_id_list, new_text_list = [], []
        dn = os.path.join(out_dir, dir_name)
        existed = set([fn for fn in os.listdir(dn) if os.stat(os.path.join(dn,fn)).st_size > 0])
        for id, text in zip(tqdm(id_list, desc='skip existed'), text_list):
            if id + '.txt' not in existed:
                new_id_list.append(id)
                new_text_list.append(text)
        return new_id_list, new_text_list
    def embed_text_parallel(texts):
        print('embed_text_parallel...')
        manager = Manager()
        return_dict = manager.dict()
        dir = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        jobs = []
        per_worker = (len(texts) + num_process - 1) // num_process
        for i in range(num_process):
            mn, mx = i * per_worker, min((i+1)*per_worker, len(texts))
            p = Process(target=embed_text_worker, args=(texts[mn:mx], max_seq_len, config, i, return_dict))
            jobs.append(p)
            p.start()
        for i in range(num_process):
            jobs[i].join()
        features = torch.cat([return_dict[i] for i in range(num_process)], dim=0)
        return features
    def save_embeddings_parallel(ids, features):
        dir = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        jobs = []
        per_worker = (len(ids) + num_process - 1) // num_process
        for i in range(num_process):
            mn, mx = i * per_worker, min((i+1)*per_worker, len(ids))
            p = Process(target=save_embeddings_worker, args=(dir_name, ids[mn:mx], features[mn:mx]))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
    ids, texts = skip_existed(ids, texts, dir_name)
    print(f'id len = {len(ids)} after skip_existed')
    if len(ids) == 0:
        return
    features = embed_text_parallel(texts)
    save_embeddings_parallel(ids, features)

def process_news():
    print('process_news')
    input = {k : [] for k in ['ids', 'titles', 'text']}
    no_content_news = {ds : [] for ds in ds_dirs}
    for ds in ds_dirs:
        for news_id in tqdm(os.listdir(ds), desc='reading ' + ds):
            content_fn = os.path.join(ds, news_id, 'news_article.json')
            with open(content_fn, 'r') as f:
                content = json.load(f)
            if len(content)==0:
                no_content_news[ds].append(news_id)
                continue
            input['ids'].append(news_id)
            input['titles'].append(content['title'])
            input['text'].append(content['text'])
        print('# no content news:', len(no_content_news[ds]))
    embed_text(input['ids'], input['titles'], max_seq_len=49, config=configs['news'], dir_name = 'news_titles')
    embed_text(input['ids'], input['text'], max_seq_len=490, config=configs['news'], dir_name = 'news_text',)

def process_tweets():
    print('process_tweets')
    with open(os.path.join(involved_dir, 'p_involved.txt'), 'r') as f:
        involved_ids_add_time = set(f.readlines()[0].strip().split(' '))
        involved_ids=set(i.split('t')[0] for i in involved_ids_add_time)
    input = {k : [] for k in ['ids', 'text']}
    for ds in ds_dirs:
        for news_id in tqdm(os.listdir(ds), 'reading ' + ds):
            tweets_dir = os.path.join(ds, news_id, 'tweets.json')
            with open(os.path.join(tweets_dir), 'r') as f:
                tweets = json.load(f)
                for tweet in tweets['tweets']:
                    if 'p' + str(tweet['tweet_id']) not in involved_ids:
                        continue
                    input['ids'].append(str(tweet["tweet_id"]))
                    input['text'].append(tweet['text'])

            retweets_dir = os.path.join(ds, news_id, 'retweets.json')
            with open(os.path.join(retweets_dir), 'r') as f:
                retweets = json.load(f)
                for tweet_id, retweet in retweets.items():
                    if len(retweet)!=0:
                        for re in retweet:
                            if 'p' + re['id_str'] not in involved_ids:
                                continue
                            input['ids'].append(re["id_str"])
                            input['text'].append(re['text'])
    embed_text(input['ids'], input['text'], max_seq_len=49, config=configs['tweet'], dir_name='tweet_text')

def process_user_worker(ds, news_id, involved_uids, return_list, i, total):
    id_desc = []
    retweet_dir = os.path.join(ds, news_id, 'retweets.json')
    with open(os.path.join(retweet_dir), 'r') as f:
        retweets=json.load(f)
        for tweet_id,retweet in retweets.items():
            if len(retweet)!=0:
                for re in retweet:
                    if 'u' + re['user']["id_str"] in involved_uids:
                        id_desc.append((re['user']["id_str"], re['user']['description']))
                    if 'u' + re["retweeted_status"]['user']["id_str"] in involved_uids:
                        id_desc.append((re["retweeted_status"]['user']["id_str"],
                                        re["retweeted_status"]['user']['description']))

    return_list.append(id_desc)
    if i % 20 == 0:
        print('process_user_worker {:7}/{:7} {:.5}'.format(i, total, i/total))

def process_user_description(num_process = 8):
    print('process_user_description')
    manager = Manager()
    with open(os.path.join(involved_dir, 'u_involved.txt'), 'r') as f:
        involved_uids_add_time = set(f.readlines()[0].strip().split(' '))
        involved_uids = set(i.split('t')[0] for i in involved_uids_add_time)
    for ds in ds_dirs:
        dir_list = os.listdir(ds)
        return_list = manager.list()
        with Pool(num_process) as p:
            p.starmap(process_user_worker, [(ds, dir_list[i], involved_uids, return_list, i, len(dir_list)) for i in range(len(dir_list))])
        ids = [i for e in return_list for (i, j) in e]
        description = [j for e in return_list for (i, j) in e]
        embed_text(ids, description, max_seq_len=49, config=configs['tweet'], dir_name='user_description', num_process=num_process)

if __name__ == '__main__':
    process_news()
    process_tweets()
    process_user_description()