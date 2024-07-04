import json
import os
from tqdm import tqdm
from multiprocessing import Manager, Pool
from datetime import datetime,timezone,timedelta


in_dir = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset'
out_dir = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def'
datasets = ['politifact']
subsets = ['fake', 'real']
num_process = 4

import shutil

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully deleted the folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting the folder: {e}")
def changetime(time_str):
    dt_object = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")

    dt_object_utc = dt_object.astimezone(timezone.utc)

    timestamp = int(dt_object_utc.timestamp())
    return str(timestamp)

def process_worker(ds_dir, news_f, return_list, i, news_ids_len):
    np_edges, pu_edges, uu_edges = [], [], []

    source_news, author_news = dict(), dict()
    source_news_t, author_news_t = dict(), dict()
    content_f = os.path.join(ds_dir, news_f, 'news_article.json')
    news_id = news_f[len("politifact"):] if news_f[:len("politifact")] == "politifact" else news_f[len("gossipcop-"):]
    news_id_t=""
    if os.path.isfile(content_f):
        try:
            with open(content_f, 'r') as f:
                content = json.load(f)
            if len(content) == 0:
                delete_folder(os.path.join(ds_dir, news_f))

            if content["source"] not in source_news.keys():
                source_news[content["source"]] = []

            if content["source"] not in source_news_t.keys():
                source_news_t[content["source"]] = []
            if content["publish_date"]:
                news_id_t=news_id+"t"+content["publish_date"]
            else:
                news_id_t = news_id + "t" +"0"
            source_news[content["source"]].append(news_id)

            source_news_t[content["source"]].append(news_id_t)

            for author in content["authors"]:
                if len(author.split()) >= 5:
                    break
                author = author.replace('About ', '')
                if author not in author_news.keys():
                    author_news[author] = []
                if author not in author_news_t.keys():
                    author_news_t[author] = []
                author_news[author].append(news_id)

                author_news_t[author].append(news_id_t)
        except Exception as e:
            print('Error reading news content:', e.__repr__(), content_f)

    tweet_f = os.path.join(ds_dir, news_f, 'tweets.json')
    if os.path.isfile(tweet_f):
        try:
            with open(tweet_f, 'r') as f:
                tweet = json.load(f)
                for tw in tweet["tweets"]:
                    np_edges.append((news_id_t, str(tw["tweet_id"])+"t"+str(tw["created_at"])))

                    pu_edges.append((str(tw["tweet_id"])+"t"+str(tw["created_at"]), str(tw['user_id'])+"t"+str(tw["created_at"])))  # 添加tweet和user边

        except Exception as e:
            print('Error reading tweet:', e.__repr__(), os.path.join(tweet_f))

    retweet_f = os.path.join(ds_dir, news_f, 'retweets.json')
    if os.path.isfile(retweet_f):
        try:
            with open(retweet_f, 'r') as f:
                retweet = json.load(f)
                for tweet_id, retws in retweet.items():
                    if len(retws) != 0:
                        for retw in retws:
                            retweet_id = retw["id_str"]+"t"+changetime(retw['created_at'])
                            forw_user = retw["user"]["id_str"]+"t"+"0"
                            orig_user = retw["retweeted_status"]["user"]["id_str"]+"t"+"0"
                            uu_edges.append((orig_user, forw_user))
                            uu_edges.append((forw_user, orig_user))
                            pu_edges.append((retweet_id, forw_user))
        except Exception as e:
            print('Error reading retweet:', e.__repr__(), os.path.join(retweet_f))

    return_list.append((np_edges, pu_edges, uu_edges, source_news, author_news, source_news_t, author_news_t))


def process(ds):
    nn_edges, np_edges, pu_edges, uu_edges = [], [], [], []
    nn_edges_t=[]
    source_news, author_news = dict(), dict()
    source_news_t, author_news_t = dict(), dict()
    for ss in subsets:
        news_ids = os.listdir(os.path.join(in_dir, ds, ss))
        manager = Manager()
        return_list = manager.list()
        ds_dir = os.path.join(in_dir, ds, ss)
        with Pool(num_process) as p:
            p.starmap(process_worker,
                      [(ds_dir, news_ids[i], return_list, i, len(news_ids)) for i in range(len(news_ids))])
        for (_np_edges, _pu_edges, _uu_edges, _source_news, _author_news, _source_news_t, _author_news_t) in return_list:
            np_edges.extend(_np_edges)
            pu_edges.extend(_pu_edges)
            uu_edges.extend(_uu_edges)
            source_news.update(_source_news)
            author_news.update(_author_news)
            source_news_t.update(_source_news_t)
            author_news_t.update(_author_news_t)

    print("for fake and real end")
    source_news_hist, author_news_hist = dict(), dict()
    for news_id_list in source_news.values():
        if len(news_id_list) not in source_news_hist.keys():
            source_news_hist[len(news_id_list)] = 0
        source_news_hist[len(news_id_list)] += 1
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges.append((nid1, nid2))
    for news_id_list in source_news_t.values():
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges_t.append((nid1, nid2))
    for news_id_list in author_news.values():
        if len(news_id_list) not in author_news_hist.keys():
            author_news_hist[len(news_id_list)] = 0
        author_news_hist[len(news_id_list)] += 1
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges.append((nid1, nid2))
    for news_id_list in author_news_t.values():
        for nid1 in news_id_list:
            for nid2 in news_id_list:
                nn_edges_t.append((nid1, nid2))

    stats = [
        '# News-news edgnt {:10} / {:10}'.format(len(set(nn_edges)), len(nn_edges)),
        '# News-news edges {:10} / {:10}'.format(len(set(nn_edges_t)), len(nn_edges_t)),
        '# News-post edges {:10} / {:10}'.format(len(set(np_edges)), len(np_edges)),
        '# Post-user edges {:10} / {:10}'.format(len(set(pu_edges)), len(pu_edges)),
        '# User-user edges {:10} / {:10}'.format(len(set(uu_edges)), len(uu_edges)),
        'source_news_hist  ' + json.dumps(source_news_hist, indent=4, sort_keys=True),
        'author_news_hist  ' + json.dumps(author_news_hist, indent=4, sort_keys=True),
    ]
    fname_dict = {
        'news-news edgnt': nn_edges,
        'news-news edges': nn_edges_t,
        'news-post edges': np_edges,
        'post-user edges': pu_edges,
        'user-user edges': uu_edges,
        'stats': [[e, ] for e in stats],
    }
    od = os.path.join(out_dir, ds)
    if not os.path.isdir(od):
        os.mkdir(od)
    for k, v in fname_dict.items():
        with open(os.path.join(od, k + '.txt'), 'w') as f:
            f.write('\n'.join([' '.join(e) for e in v]) + '\n')


if __name__ == '__main__':
    for dataset in datasets:
        process(dataset)