from typing import List, Tuple, Optional
import torch
import os
from transformers import XLMRobertaModel, XLMRobertaTokenizer, AutoModel, AutoTokenizer,AutoModelForSeq2SeqLM,T5ForConditionalGeneration,T5Tokenizer


class TextEmbedder(torch.nn.Module):
    def __init__(self, max_seq_len : int, model_name: str, model_path: Optional[str] = '', device: Optional[str] = 'cpu'):
        super(TextEmbedder, self).__init__()
        """
        Parameters
        ----------
        max_len : int

        model_name : str
            The name of the model, should be one of 'word2vec', 'xlm-roberta-base', 'xlm-roberta-large', 'vinai/bertweet-base', 'mrm8488/t5-base-finetuned-summarize-news', 'jordan-m-young/buzz-article-gpt-2'
        model_path : str, optional
            The path to the w2v file / finetuned Transformer model path. Required for w2v.
        """
        
        assert model_name in ['word2vec', 'xlm-roberta-base', 'xlm-roberta-large', 'bertweet-base', 't5-base-finetuned-summarize-news', 'jordan-m-young/buzz-article-gpt-2']
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self.device = torch.device(device)
        if model_path == '':
            model_path = model_name  # TODO check if the 
        print('TextEmbedder: Loading model {} ({})'.format(model_name, model_path))
        if model_name == 'word2vec':
            assert os.path.isfile(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', model_max_length=self.max_seq_len+1)
            self._load_weibo_w2v(model_path)
            self.embed_dim = 300
        elif model_name in ['bertweet-base',  'jordan-m-young/buzz-article-gpt-2', 'jordan-m-young/buzz-article-gpt-2']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = AutoModel.from_pretrained(model_path, return_dict=True).to(self.device)  # T5 for news doesn't have 'add_pooling_layer' option
            self.embed_dim = 768
        elif model_name in ['t5-base-finetuned-summarize-news']:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, return_dict=True).to(
                self.device)  # T5 for news doesn't have 'add_pooling_layer' option
            self.embed_dim = 768
        else:
            assert model_path in ['xlm-roberta-base', 'xlm-roberta-large'] or os.path.isdir(model_path)
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, model_max_length=self.max_seq_len)
            self.model = XLMRobertaModel.from_pretrained(model_path, return_dict=True, add_pooling_layer=False).to(self.device)
            self.embed_dim = 768
        print('TextEmbedder: Finished loading model {}'.format(model_name))

    def forward(self, text_list: List[str], return_tokens: Optional[bool] = False) -> Tuple[torch.Tensor, List[List[str]]]:
        """Embeds a list of text into a torch.Tensor

        Parameters
        ----------
        text_list : List[str]
            Each item is a piece of text
        return_tokens: Optional[bool] = False
            For debug only. It slows down everything if you're using a Transformer, so don't use it unless necessary.

        Returns 
        ----------
        outputs: torch.Tensor
            Embedding of shape (len(text_list), max_seq_len, embed_dim)
        tokens: List[List[str]]
            (if return_tokens=True) A list of tokenized original text (padded for Transformers; not padded for w2v)
        """

        if self.model_name == 'word2vec':
            tokens = [self.tokenizer.tokenize(text)[1: 1 + self.max_seq_len] for text in text_list]
            tokens = [[token.replace('▁', '') for token in doc] for doc in tokens]  # NOTE '▁' and '_' (underscore) are different
            outputs = self._w2v_embed(tokens)
        elif self.model_name =='t5-base-finetuned-summarize-news':
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_seq_len, padding='max_length',
                                    truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs,decoder_input_ids=inputs['input_ids'])['encoder_last_hidden_state'].detach().cpu()
            if return_tokens:
                tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_seq_len, padding='max_length', truncation=True)
            inputs = {k : v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)['last_hidden_state'].detach().cpu()
            if return_tokens:
                tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids']]
        assert outputs.shape == (len(text_list), self.max_seq_len, self.embed_dim)
        return (outputs, tokens) if return_tokens else outputs

    def compute_seq_len_statistics(text_list: List[str], config):
        from tqdm import tqdm
        import numpy as np
        import json
        tokenizer = AutoTokenizer.from_pretrained(config['model name'])
        bsz = config['batch size']
        n_batches = (len(text_list) + bsz - 1) // bsz
        counts = []
        for i in tqdm(range(n_batches)):
            mn, mx = i * bsz, min(len(text_list), (i + 1) * bsz)
            inputs = tokenizer(text_list[mn:mx])['input_ids']
            counts.extend([len(i) for i in inputs])
        stats = {
            'mean' : np.mean(counts),
            'std' : np.std(counts),
            'max' : max(counts),
            'min' : min(counts),
        }
        print(json.dumps({'tweet_stat' : stats}))
        return stats

    def _load_weibo_w2v(self, model_path):
        self.w2v = dict()
        with open(model_path) as w2v_file:
            lines = w2v_file.readlines()
        info, vecs = lines[0], lines[1:]
        
        info = info.strip().split()
        self.vocab_size, self.embed_dim = int(info[0]), int(info[1])

        for vec in vecs:
            vec = vec.strip().split()
            self.w2v[vec[0]] = [float(val) for val in vec[1:]]

    def _w2v_embed(self, docs):
        outputs = torch.zeros((len(docs), self.max_seq_len, self.embed_dim))  # no valid words => zeros
        for i, doc in enumerate(docs):
            for j, token in enumerate(doc[:self.max_seq_len]):
                outputs[i, j] = torch.tensor(self.w2v.get(token, 0))
        return outputs


if __name__ == '__main__':
    text_list_weibo = ['酷！艾薇儿现场超强翻唱Ke$ha神曲TiK ToK！超爱这个编曲！ http://t.cn/htjA04', '转发微博。']
    text_list_fnn_tweet = ["@Calila1988 No. I hate Brad Pitt. B.J. Novak is way cooler. I know this because he is Huma's secret lover.", "I've always loved Brad Pitt, he's my secret lover ;) Sorry Angelina. No longer Brangelina, now i'ts Bralde ;) hahaha"]
    text_list_fnn_news = ["Star magazine has released an explosive report today that claims a woman has come forward to claim she is pregnant with Brad Pitt's child.\n\nThe 54-year-old actor has allegedly learnt that a former 'twenty-something' fling from earlier this year, who wishes to remain anonymous, has come forward to the publication.\n\n'This will be an absolute nightmare for Bard if her claims are true,' an insider spilled. 'After all the drama he's been through over the past two years, he's desperate to keep his life as trouble and scandal free as possible.'\n\nAccording to Star's bombshell claims, the mystery woman is willing to undergo a paternity test and use the results to effectively tell Brad, 'I've got the DNA tests to prove it!'","MAY 02, 2017\n\nRDR Staff\n\nAbout prosopagnosia\n\nAbout epidermolysis bullosa\n\nStay informed on the latest rare disease news and developments by signing up for our newsletter\n\nIn this week\u2019s Hollywood gossip, it was revealed that Brad Pitt does not have a rare disease that is \u2018eating him away.\u2019Whew.However, the actor is no stranger to the rare disease community. Pitt is an honorary member of the Epidermolysis Bullosa Medical Research Foundation (EBMRF) that is dedicated to raising money and awareness for the rare skin disorder, epidermolysis bullosa.In January, Pitt introduced singer Chris Cornell during a Rock4EB annual charity to support EBMRF.Further, he revealed in a 2013 Esquire Magazine interview that he self-diagnosed himself with a rare disease \u2013 prosopagnosia \u2014 also known as face blindness. According to the article, \"Brad Pitt won't remember you. If you've met him, he'll have no idea who you are when he meets you again. Even if you've had what he calls \"a real conversation,\" your face will start fading from his memory as soon as you walk away.\"It should be noted that being diagnosed with a condition and being self-diagnosed with a condition can be 2 very different diagnoses.Prosopagnosia refers to the inability to recognize familiar people from their face. They may also be unable to recognize other stimuli, such as objects, cars, or animals.People with prosopagnosia typically learn to use non-face cues including voice, walking style and hairstyle to recognize others. The underlying genetic cause of developmental prosopagnosia is not known but familial reports of this condition are consistent with autosomal dominant inheritance.Epidermolysis bullosa (aka 'the worst disease you've never heard of') is a group of devastating, life-threatening genetic skin disorders impacting children that is characterized by skin blisters and erosions all over the body.The most severe form, recessive dystrophic epidermolysis bullosa, is characterized by chronic skin blistering, open and painful wounds, joint contractures, esophageal strictures, pseudosyndactyly, corneal abrasions and a shortened life span. Patients with recessive dystrophic epidermolysis bullosa lack functional type VII collagen (C7) owing to mutations in the gene COL7A1 that encodes for C7 and is the main component of anchoring fibrils that attach the dermis to the epidermis.Epidermolysis bullosa patients suffer through intense pain throughout their lives, with no effective treatments available to reduce the severity of their symptoms."]
    max_seq_len = 49
    '''
    model_name="bertweet-base"
    model=AutoModel.from_pretrained(model_name)
    model.save_pretrained("vinai/bertweet-base")
    '''
    # embedder = TextEmbedder(max_seq_len, 'xlm-roberta-base', finetuned_transformer_path)
    # outputs, tokens = embedder(text_list_weibo, return_tokens=True)
    # print(tokens)
    # print(outputs.shape)

    # embedder = TextEmbedder(max_seq_len, 'word2vec', word2vec_path)
    # outputs, tokens = embedder(text_list_weibo, return_tokens=True)
    # print(tokens)
    # print(outputs.shape)

    # embedder = TextEmbedder(max_seq_len, 'bertweet-base')
    # outputs, tokens = embedder(text_list_fnn_tweet, return_tokens=True)
    # print(tokens)
    # print(outputs.shape)

    embedder = TextEmbedder(max_seq_len, 't5-base-finetuned-summarize-news')
    outputs = embedder(text_list_fnn_news)
    # print(tokens)
    print(outputs)
