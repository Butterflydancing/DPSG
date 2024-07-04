import os
import json
from tqdm import tqdm

ds_path = 'FakeNewsNet-Dataset/FakeNewsNet_Dataset'
datasets = ['politifact_fake','politifact_real', 'gossipcop_fake', 'gossipcop_real']

keys = [
    '# News articles',
    '# News with images downloaded',
    '# Users posting tweets',
    '# Tweets posting news', 
    '# Tweets with retweets',
    '# News with retweets', '# News with retweets downloaded',
    '# News with tweets', '# News with tweets downloaded',
    '# News images processed'
]

uids = set()
tweets_with_retweets = set()
all_stats = {k1 : {} for k1 in datasets}


goimgs=set()
poimgs=set()
goimg_list=os.listdir('FakeNewsNet-Dataset/NewsImages/gossipcop_images')
poimg_list=os.listdir('FakeNewsNet-Dataset/NewsImages/politifact_images')
for goimg in tqdm(goimg_list,desc="goimg_list"):
    goimg_nid=goimg.split(".")[0]
    if goimg_nid == '.DS_Store': continue
    goimgs.add(goimg_nid)
for poimg in tqdm(poimg_list,desc="poimg_list"):
    poimg_nid=poimg.split(".")[0]
    if poimg_nid == '.DS_Store': continue
    poimgs.add(poimg_nid)

for ds in datasets:
    print(ds,  'starts')
    stats = {key : 0 for key in keys}
    uids = set()
    img_processed_nid = set()
    news_list = os.listdir(os.path.join(ds_path, ds))
    stats['# News articles'] += len(news_list)
    for nidpath in tqdm(news_list, desc=ds):
        if nidpath[:len("politifact")]=="politifact":
            nid=nidpath[len("politifact"):]
        else:
            nid=nidpath[len("gossipcop-"):]
        if nid in goimgs or nid in poimgs:
            img_processed_nid.add(nid)
        news_content_path = os.path.join(ds_path, ds, nidpath, 'news_content.json')
        tweet_path = os.path.join(ds_path, ds, nidpath, 'tweets.json')
        retweet_path = os.path.join(ds_path, ds, nidpath, 'retweets.json')
        if os.path.isfile(tweet_path):
            stats['# News with tweets downloaded'] += 1
            with open(tweet_path,'r') as tf:
                tweet=json.load(tf)
                stats['# Tweets posting news'] += len(tweet['tweets'])
                for tweet_f in tweet['tweets']:
                    uids.add(tweet_f['user_id'])
        if os.path.isfile(retweet_path):
            stats['# News with retweets'] += 1
            with open(retweet_path,'r') as rtf:
                retweet=json.load(rtf)
                for tweet_id,re in retweet.items():
                    if len(retweet[tweet_id])!=0:
                        stats['# News with retweets downloaded'] += 1
                        tweets_with_retweets.add(tweet_id)
                        for rt in retweet[tweet_id]:
                            uids.add(rt["user"]["id"])
    stats['# Users posting tweets'] = len(uids)
    stats['# Tweets with retweets'] = len(tweets_with_retweets)
    stats['# News images processed'] = len(img_processed_nid)
    print(ds, 'ends')
    all_stats[ds] = stats

print('#' * 10 + ' Overall Stats ' + '#' * 10)

print('\\toprule')

print(' ' * 40, end = "")
for ds in datasets:
    print(' & {:10} & {:10}'.format(ds, ds), end='')


print('\\\\')
print(' ' * 40, end = "")
for i in range(2):
    print(' & {:10} & {:10}'.format('Fake', 'Real'), end='')

    
print('\\\\')

print('\\midrule')

for k in keys:
    print('{:40}'.format(k.replace('#', "\\#")), end='')
    for ds in datasets:
        # for ss in subset:
        print(' & {:10}'.format(int(all_stats[ds][k])), end='')
    print('\\\\')

print('\\bottomrule')
    