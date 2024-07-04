#!/usr/bin/env python
# coding: utf-8


import numpy as np
import torch
import torch.nn as nn
# import torch.nn.init as init
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable
import math
from torch_position_embedding import PositionEmbedding
import random
import os
import time
import json
import scipy.sparse as sp
import torch.nn.functional as F
import torch.nn.init as init

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


p_add_time_dict={}
n_add_time_dict={}
with open("../../data/rwr_results/fnn_gossipcop_u50/n_add_time.txt",'r') as f:
    for line in f:
        line = line.strip()
        if line:
            key, value = line.split(' ', 1)
            n_add_time_dict['gossipcop-'+key] = value
with open("../../data/rwr_results/fnn_gossipcop_u50/p_add_time.txt",'r') as f:
    for line in f:
        line = line.strip()
        if line:
            key, value = line.split(' ', 1)
            p_add_time_dict[key] = value
class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_news=[], neighbor_list_post=[], neighbor_list_user=[],
                 label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label
        self.neighbors_news = neighbor_list_news
        self.neighbors_post = neighbor_list_post
        self.neighbors_user = neighbor_list_user

def neighbor_loader(pathway):
    neighbor_dict_post = {}
    neighbor_dict_user = {}
    neighbor_dict_news = {}
    neighbor_dict_n_p_u = {}
    neighbor_dict_n_p_u_add_time={}
    neighbor_dict_n = {}
    f = open(pathway)
    Lines = f.readlines()

    for i in range(len(Lines)):
        neighbor_list = Lines[i].split()
        for item in neighbor_list[1:]:
            node_type = item[0]
            node_id = item[1:]

            if node_type == 'n' and item[1:] != 'PADDING':
                if node_id in n_add_time_dict:
                    time_value = n_add_time_dict[node_id]
                else:
                    time_value = '0'

            elif node_type == 'p' and item[1:] != 'PADDING':
                if node_id in p_add_time_dict:
                    time_value = p_add_time_dict[node_id]
                else:
                    time_value = '0'

            elif node_type == 'u' and item[1:] != 'PADDING':
                time_value = "0"


            if neighbor_list[0][1:-1] not in neighbor_dict_n_p_u_add_time:
                neighbor_dict_n_p_u_add_time[neighbor_list[0][1:-1]] = []

            neighbor_dict_n_p_u_add_time[neighbor_list[0][1:-1]].append(time_value)

        neighbor_dict_n_p_u[(neighbor_list[0][1:-1]).split("t")[0]] = [(item[0], (item[1:]).split("t")[0]) for item in
                                                                       neighbor_list[1:] if item[1:] != 'PADDING']

        neighbor_dict_n[(neighbor_list[0][1:-1]).split("t")[0]] = [(item[0], (item[1:]).split("t")[0]) for item in
                                                                   neighbor_list[1:] if
                                                                   item[0] == 'n' and item[1:] != 'PADDING']

        neighbor_dict_news[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'n' and item[1:]!='PADDING']
        neighbor_dict_user[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'u' and item[1:]!='PADDING']
        neighbor_dict_post[neighbor_list[0][1:-1]] = [item[1:] for item in neighbor_list[1:] if item[0] == 'p' and item[1:]!='PADDING']
    return neighbor_dict_n_p_u, neighbor_dict_n,neighbor_dict_n_p_u_add_time,neighbor_dict_news, neighbor_dict_post, neighbor_dict_user

neighbor_dict = neighbor_loader('../../data/rwr_results/fnn_gossipcop_u50/n_neighbors.txt')

with open("../../data/FakeNewsNet-Dataset/FakeNewsNet_Dataset/graph_def/gossipcop/original_adj",'r') as f:
    adj_list=json.load(f)
    print(len(adj_list))

def data_loader(pathway,node_type):
    if node_type == "news":
        news_node = []
        news_id = []
        news_label = []
        news_embed = []
        news_n_neigh = []
        news_p_neigh = []
        news_u_neigh = []
        for i in range(len(os.listdir(pathway))):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j < len(Lines):
                    if j % 7 == 0:
                        _, id_, label = Lines[j].split()
                        news_id.append(id_)
                        news_label.append(int(label))
                        embed = []
                    if j % 7 == 1 or j % 7 == 2 or j % 7 == 3:
                        embed.append(list(map(float, Lines[j].split())))
                    if j % 7 == 3:
                        news_embed.append(embed)
                    if j % 7 == 4:
                        n_neigh = Lines[j].split()
                        news_n_neigh.append(n_neigh)
                    if j % 7 == 5:
                        p_neigh = Lines[j].split()
                        news_p_neigh.append(p_neigh)
                    if j % 7 == 6:
                        u_neigh = Lines[j].split()
                        news_u_neigh.append(u_neigh)
            f.close()
        for i in range(len(news_id)):
            node = Het_Node(node_type="news", node_id=news_id[i], embed=news_embed[i],
                            neighbor_list_news=news_n_neigh[i], neighbor_list_post=news_p_neigh[i],
                            neighbor_list_user=news_u_neigh[i], label=news_label[i])
            news_node.append(node)
        return news_node

    elif node_type == 'post':
        post_node = []
        post_id = []
        post_embed = []
        for i in range(len(os.listdir(pathway))):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j < len(Lines):
                    if j % 6 == 0:
                        id_ = Lines[j].split()
                        post_id.append(id_[1])
                        embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        embed.append(list(map(float, Lines[j].split())))
                    if j % 6 == 2:
                        post_embed.append(embed)
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type="post", node_id=post_id[i], embed=post_embed[i])
            post_node.append(node)
        return post_node

    else:
        user_node = []
        user_id = []
        user_embed = []
        for i in range(len(os.listdir(pathway))):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines()
            for j in range(len(Lines)):
                if j < len(Lines):
                    if j % 6 == 0:
                        id_ = Lines[j].split()
                        user_id.append(id_[1])
                        embed = []
                    if j % 6 == 1 or j % 6 == 2:
                        embed.append(list(map(float, Lines[j].split())))
                    if j % 6 == 2:
                        user_embed.append(embed)
            f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type="user", node_id=user_id[i], embed=user_embed[i])
            user_node.append(node)
        return user_node
post_nodes = data_loader(pathway='../../data/processed_data/FakeNewsNet/GossipCop/fnn_gossipcop_50/normalized_post_nodes/', node_type="post")
user_nodes = data_loader(pathway='../../data/processed_data/FakeNewsNet/GossipCop/fnn_gossipcop_50/normalized_user_nodes/', node_type="user")
news_nodes = data_loader(pathway='../../data/processed_data/FakeNewsNet/GossipCop/fnn_gossipcop_50/normalized_news_nodes/', node_type="news")

news_emb_dict = {}
post_emb_dict = {}
user_emb_dict = {}

for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb
for news in news_nodes:
    news_emb_dict[news.node_id] = news.emb

news_nodes_real = []
news_nodes_fake = []
for node in news_nodes:
    if node.label == 1:
        news_nodes_real.append(node)
    else:
        news_nodes_fake.append(node)
ratio = len(news_nodes_real) / len(news_nodes_fake)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.wtrans = nn.Parameter(torch.zeros(size=(2 * out_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.wtrans.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)
        N = h.size()[0]
        Wh1 = torch.mm(h, self.a[:self.out_features, :])
        Wh2 = torch.mm(h, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.T)
        zero_vec = -1e12 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        negative_attention = torch.where(adj > 0, -e, zero_vec)
        attention = F.softmax(attention, dim=1)
        negative_attention = -F.softmax(negative_attention,dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        negative_attention = F.dropout(negative_attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, inp)
        h_prime_negative = torch.matmul(negative_attention, inp)
        h_prime_double = torch.cat([h_prime,h_prime_negative],dim=1)
        new_h_prime = torch.mm(h_prime_double,self.wtrans)

        if self.concat:
            return F.elu(new_h_prime)
        else:
            return new_h_prime

class Signed_GAT(nn.Module):
    def __init__(self,nb_heads = 1, dropout = 0, alpha = 0.3):
        super(Signed_GAT, self).__init__()
        self.dropout = dropout
        self.nb_heads = nb_heads
        self.alpha = alpha

    def forward(self,node_embedding,cosmatrix,original_adj, X_tid):
        embedding_dim=node_embedding.shape[1]
        node_num=original_adj.shape[0]

        user_tweet_embedding = nn.Embedding(num_embeddings=node_num, embedding_dim=embedding_dim, padding_idx=0).to(device)
        user_tweet_embedding.from_pretrained(node_embedding)

        original_adj = torch.from_numpy(original_adj.astype(np.float64)).to(device)
        potentinal_adj = torch.where(cosmatrix > 0.5, torch.ones_like(cosmatrix), torch.zeros_like(cosmatrix)).to(device)
        adj = original_adj + potentinal_adj
        adj = torch.where(adj > 0, torch.ones_like(adj), torch.zeros_like(adj))

        self.attentions = [
            GraphAttentionLayer(embedding_dim, embedding_dim, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in
            range(self.nb_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        out_att = GraphAttentionLayer(embedding_dim * self.nb_heads, embedding_dim, dropout=self.dropout, alpha=self.alpha,
                                           concat=False)

        X = user_tweet_embedding(torch.arange(0, node_num).long().to(device)).to(torch.float32)
        x = F.dropout(X, self.dropout, training=self.training)
        adj = adj.to(torch.float32)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.sigmoid(out_att(x, adj))
        return x,x[X_tid]

class TransformerBlock(nn.Module):

    def __init__(self, input_size, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)
        Q_K_score = self.dropout(Q_K_score)

        V_att = Q_K_score.bmm(V)
        return V_att


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att = self.scaled_dot_product_attention(Q_, K_, V_)
        V_att = V_att.view(bsz, self.n_heads, q_len, self.d_v)
        V_att = V_att.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)

        output = self.dropout(V_att.matmul(self.W_o))
        return output


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, H)
        :param K: (batch_size, max_k_words, H)
        :param V: (batch_size, max_v_words, H)
        :return:  output: (batch_size, max_q_words, H)  same size as Q
        '''
        V_att = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output

class DPSG(nn.Module):
    def __init__(self,input_dim, n_hidden_dim, u_hidden_dim,p_hidden_dim,
                 out_embed_d, d_model=512,attn_heads=2,enc_layers=1,outemb_d=1,
                 content_dict={},npu=30, self_attn_heads=2):
        super(DPSG, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = out_embed_d
        self.embed_d = out_embed_d
        self.n_input_dim = out_embed_d
        self.n_hidden_dim = n_hidden_dim
        self.n_output_dim = out_embed_d
        self.u_input_dim = out_embed_d
        self.u_hidden_dim = u_hidden_dim
        self.u_output_dim = out_embed_d
        self.p_input_dim = out_embed_d
        self.p_hidden_dim = p_hidden_dim
        self.p_output_dim = out_embed_d
        self.d_model = d_model
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        self.content_dict = content_dict
        self.npu = npu
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=200)
        self.pos_decoder = PositionalEncoding(d_model, dropout=0.1, max_len=200)
        self.type_encoder = nn.Embedding(3, d_model, padding_idx=0)
        self.n_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.register_parameter('bias', None)
        self.transformer = nn.Transformer(d_model=d_model, nhead=attn_heads, num_encoder_layers=enc_layers,
                                          num_decoder_layers=1, dim_feedforward=512,
                                          dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)

        self.init_linear_text = nn.Linear(self.input_dim[0], self.hidden_dim)
        self.init_linear_image = nn.Linear(self.input_dim[1], self.hidden_dim)
        self.init_linear_time = nn.Linear(1, self.hidden_dim)
        self.init_linear_other_p = nn.Linear(self.input_dim[2], self.hidden_dim)
        self.init_linear_other_user = nn.Linear(self.input_dim[3], self.hidden_dim)

        self.post_content_attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.news_title_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.news_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.post_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.user_content_attention_text = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_image = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_time = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.attention_other = nn.MultiheadAttention(self.hidden_dim, self_attn_heads, dropout=0.2)
        self.layernorm1 = nn.LayerNorm([1, out_embed_d])
        self.layernorm2 = nn.LayerNorm([1, out_embed_d])
        self.layernorm3 = nn.LayerNorm([1, out_embed_d])
        self.layernorm4 = nn.LayerNorm([1, out_embed_d])
        self.layernorm5 = nn.LayerNorm([1, out_embed_d])
        self.layernorm6 = nn.LayerNorm([1, out_embed_d])

        self.n_init_linear = nn.Linear(self.n_input_dim, self.n_hidden_dim)
        self.n_attention = nn.MultiheadAttention(self.n_hidden_dim, self_attn_heads, dropout=0.2)
        self.n_linear = nn.Linear(self.n_hidden_dim, self.n_output_dim)

        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_hidden_dim)
        self.u_attention = nn.MultiheadAttention(self.u_hidden_dim, self_attn_heads, dropout=0.2)
        self.u_linear = nn.Linear(self.u_hidden_dim, self.u_output_dim)

        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_hidden_dim)
        self.p_attention = nn.MultiheadAttention(self.p_hidden_dim, self_attn_heads, dropout=0.2)
        self.p_linear = nn.Linear(self.p_hidden_dim, self.p_output_dim)
        self.act = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.25)
        self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d)

        self.output_act = nn.Sigmoid()

        self.mh_attention1 = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0)
        self.mh_attention = TransformerBlock(input_size=self.hidden_dim, n_heads=8, attn_dropout=0,is_layer_norm=True)

        self.alignfc_g = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.alignfc_c = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.fc3 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.6)
        self.gat_relation = Signed_GAT()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def build_symmetric_adjacency_matrix(self,edges, shape):
        def normalize_adj(mx):
            rowsum = np.array(mx.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        adj = sp.coo_matrix(arg1=(edges[:, 2], (edges[:, 0], edges[:, 1])), shape=shape,
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj.tocoo()

    def calculate_cos_matrix(self,node_embedding):
        a,b = node_embedding,node_embedding.T
        c = torch.mm(a, b)
        aa = torch.mul(a, a)
        bb = torch.mul(b, b)
        asum = torch.sqrt(torch.sum(aa, dim=1, keepdim=True))
        bsum = torch.sqrt(torch.sum(bb, dim=0, keepdim=True))
        norm = torch.mm(asum, bsum)
        res = torch.div(c, norm)
        return res

    def Bi_RNN(self, neighbor_id, neighbor_time,node_type, post_emb_dict, user_emb_dict, news_emb_dict):
        new_id = []
        if node_type == "news":
            input_title = []
            input_content = []
            input_image = []
            for i in neighbor_id:
                if ("news", i) not in self.content_dict:
                    input_title.append(news_emb_dict[i][0])
                    input_content.append(news_emb_dict[i][1])
                    input_image.append(news_emb_dict[i][2])
                    new_id.append(i)
            input_title = torch.Tensor(input_title).to(device)
            input_image = torch.Tensor(input_image).to(device)
            input_content = torch.Tensor(input_content).to(device)
            input_time = torch.Tensor(neighbor_time).to(device)
            input_time = input_time.view(-1, 1)

            linear_input_title = self.init_linear_text(input_title)
            linear_input_time=self.init_linear_time(input_time)
            linear_input_content = self.init_linear_text(input_content)
            linear_input_image = self.init_linear_image(input_image)

            linear_input_title = linear_input_title.view(linear_input_title.shape[0], 1, linear_input_title.shape[1])
            linear_input_content = linear_input_content.view(linear_input_content.shape[0], 1,linear_input_content.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            linear_input_time = linear_input_time.view(linear_input_time.shape[0], 1,linear_input_time.shape[1])

            attention_out_title, self.hidden_text = self.news_title_attention_text(linear_input_title,linear_input_title,linear_input_title)
            attention_out_content, self.hidden_text = self.news_content_attention_text(linear_input_content,linear_input_content,linear_input_content)
            attention_out_image, self.hidden_image = self.attention_image(linear_input_image, linear_input_image,linear_input_image)
            attention_out_time, self.hidden_time = self.attention_time(linear_input_time, linear_input_time,linear_input_time)

            attention_out_title = self.layernorm1(attention_out_title)
            attention_out_content = self.layernorm2(attention_out_content)
            attention_out_time = self.layernorm6(attention_out_time)

            co_att_tt = self.mh_attention(attention_out_time, attention_out_title, attention_out_title)
            co_att_tc = self.mh_attention(attention_out_time, attention_out_content, attention_out_content)
            co_att_ti = self.mh_attention(attention_out_time, attention_out_image, attention_out_image)

            concate = torch.cat((co_att_tt, co_att_tc, co_att_ti), 1)
        if node_type == "post":
            input_a = []
            input_b = []
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(post_emb_dict[i][1])
                    input_b.append(post_emb_dict[i][0])
                    new_id.append(i)
            input_a = torch.Tensor(input_a).to(device)
            input_b = torch.Tensor(input_b).to(device)
            input_time = torch.Tensor(neighbor_time).to(device)
            input_time = input_time.view(-1, 1)
            linear_input_text = self.init_linear_text(input_a)
            linear_input_other = self.init_linear_other_p(input_b)
            linear_input_time = self.init_linear_time(input_time)

            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            linear_input_time = linear_input_time.view(linear_input_time.shape[0], 1, linear_input_time.shape[1])

            attention_out_text, self.hidden_text = self.post_content_attention_text(linear_input_text,linear_input_text,linear_input_text)
            attention_out_other, self.hidden_other = self.post_content_attention_other(linear_input_other,linear_input_other,linear_input_other)
            attention_out_time, self.hidden_time = self.attention_time(linear_input_time, linear_input_time,linear_input_time)

            attention_out_text = self.layernorm3(attention_out_text)
            attention_out_other = self.layernorm3(attention_out_other)
            attention_out_time = self.layernorm6(attention_out_time)

            co_att_tt = self.mh_attention(attention_out_time, attention_out_text, attention_out_text)
            co_att_to = self.mh_attention(attention_out_time, attention_out_other, attention_out_other)

            concate = torch.cat((co_att_tt, co_att_to), 1)
        if node_type == "user":
            input_a = []
            input_b = []
            for i in neighbor_id:
                if ("user", i) not in self.content_dict:
                    input_a.append(user_emb_dict[i][0])
                    input_b.append(user_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a).to(device)
            input_b = torch.Tensor(input_b).to(device)

            linear_input_text = self.init_linear_text(input_b)
            linear_input_other = self.init_linear_other_user(input_a)

            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])

            attention_out_text, self.hidden_text = self.user_content_attention_text(linear_input_text,linear_input_text,linear_input_text)
            attention_out_other, self.hidden_other = self.attention_other(linear_input_other, linear_input_other,linear_input_other)

            attention_out_text = self.layernorm4(attention_out_text)
            attention_out_other = self.layernorm5(attention_out_other)

            concate = torch.cat((attention_out_text, attention_out_other), 1)
        mean_pooling = torch.mean(concate, 1)
        return mean_pooling

    def SameType_Agg_Bi_RNN(self, neighbor_id,neighbor_time, node_type):
        content_embedings = self.Bi_RNN(neighbor_id,neighbor_time, node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        aft_rnn_dict = {}
        for i in range(len(neighbor_id)):
            aft_rnn_dict[neighbor_id[i]] = content_embedings[i]
        return aft_rnn_dict

    def transformer_agg(self, het_node, neighbor_order_n_p_u,all_t0,npu=30):

        all_t = [float(i.split('t')[-1]) for i in all_t0]
        all_tt = torch.tensor(all_t[1:]) - all_t[0]
        all_t_normalized = (all_tt - all_tt.min()) / max((all_tt.max() - all_tt.min()), 1e-8)

        all_tt_exp = torch.exp(-all_t_normalized).to(device)
        all_tt_exp = (all_tt_exp - all_tt_exp.min()) / max((all_tt_exp.max() - all_tt_exp.min()), 1e-8)

        node_embedding = []
        c_agg_batch = self.Bi_RNN([het_node.node_id], [0],het_node.node_type, post_emb_dict, user_emb_dict, news_emb_dict)
        node_embedding.append(c_agg_batch[0])
        node_embedding_n = []
        node_embedding_n.append(c_agg_batch[0])

        now_node = []
        now_node.append("n" + het_node.node_id)
        now_node_n = []
        now_node_n.append("n" + het_node.node_id)

        post_neighbor = []
        post_time = []
        user_neighbor = []
        user_time = []
        news_neighbor = []
        news_time = []
        for item in neighbor_order_n_p_u[:npu]:
            if item[0] == 'p':
                post_neighbor.append(item[1])
                index = neighbor_order_n_p_u.index(('p', item[1]))
                post_time.append(all_tt_exp[index])
            elif item[0] == 'u':
                user_neighbor.append(item[1])
                index = neighbor_order_n_p_u.index(('u', item[1]))
                user_time.append(all_tt_exp[index])
            elif item[0] == 'n':
                news_neighbor.append(item[1])
                index = neighbor_order_n_p_u.index(('n', item[1]))
                news_time.append(all_tt_exp[index])

        if news_neighbor:
            n_aft_rnn_dict = self.SameType_Agg_Bi_RNN(news_neighbor, news_time,"news")
            for id,value in n_aft_rnn_dict.items():
                now_node.append("n"+id)
                now_node_n.append("n"+id)
                node_embedding.append(value)
                node_embedding_n.append(value)
        if user_neighbor:
            u_aft_rnn_dict = self.SameType_Agg_Bi_RNN(user_neighbor, user_time,"user")
            for id,value in u_aft_rnn_dict.items():
                now_node.append("u" + id)
                node_embedding.append(value)
        if post_neighbor:
            p_aft_rnn_dict = self.SameType_Agg_Bi_RNN(post_neighbor, post_time,"post")
            for id,value in p_aft_rnn_dict.items():
                now_node.append("p" + id)
                node_embedding.append(value)
        node_embedding = torch.stack(node_embedding)
        node_embedding_n = torch.stack(node_embedding_n)


        now_adj_list_old = {}
        for node in now_node:
            try:
                c = list(set(adj_list[node]) & set(now_node))
            except KeyError:
                c = list(set(now_node))
            if len(c) != 0:
                now_adj_list_old[node] = c

        node_id2_xulie = {id: i for i, id in enumerate(now_node)}
        now_node_n = [node_id2_xulie[i] for i in now_node_n]
        num_node = len(node_id2_xulie)
        now_adj_list = {}
        for id, v in now_adj_list_old.items():
            i = node_id2_xulie[id]
            now_adj_list[i] = []
            for jid in v:
                j = node_id2_xulie[jid]
                now_adj_list[i].append(j)

        relation = []
        for id, v in now_adj_list.items():
            for dst in v:
                relation.append([id, dst, '1'])
        relation = np.array(relation)

        adj = self.build_symmetric_adjacency_matrix(edges=relation, shape=(num_node, num_node))

        original_adj = np.zeros(shape=adj.shape)
        for i, v in now_adj_list.items():
            v = [int(e) for e in v]
            original_adj[int(i), v] = 1

        self.cosmatrix = self.calculate_cos_matrix(node_embedding)

        gx,graph_feature = self.gat_relation.forward(node_embedding=node_embedding,
                                                  cosmatrix=self.cosmatrix,
                                                  original_adj=original_adj, X_tid=now_node_n)

        graph_feature_all = []
        graph_feature_all.append(gx[0])
        for type_id in neighbor_order_n_p_u[:npu]:
            gg = gx[now_node.index(type_id[0] + type_id[1])]
            graph_feature_all.append(gg)
        graph_feature_all = torch.stack(graph_feature_all)

        bsz = len(now_node_n)
        bsz_all = len(neighbor_order_n_p_u[:npu]) + 1
        self_att_g = self.mh_attention1(graph_feature.view(bsz, -1, self.hidden_dim),graph_feature.view(bsz, -1, self.hidden_dim), graph_feature.view(bsz, -1, self.hidden_dim))
        self_att_c = self.mh_attention1(node_embedding_n.view(bsz, -1, self.hidden_dim),node_embedding_n.view(bsz, -1, self.hidden_dim), \
                                       node_embedding_n.view(bsz, -1, self.hidden_dim))
        self_att_g_all = self.mh_attention1(graph_feature_all.view(bsz_all, -1, self.hidden_dim),graph_feature_all.view(bsz_all, -1, self.hidden_dim), \
                                            graph_feature_all.view(bsz_all, -1,self.hidden_dim))

        align_c = self.alignfc_c(self_att_c).view(bsz, self.hidden_dim)
        align_g = self.alignfc_g(self_att_g).view(bsz, self.hidden_dim)
        align_g_all = self.alignfc_g(self_att_g_all).view(bsz_all, self.hidden_dim)

        dist = [align_c, align_g]

        encoder_emb_input = []
        decoder_emb_input = []
        encoder_order_input = []
        decoder_type_input = []
        decoder_order_input = []
        encoder_type_input = []
        encoder_order_input.append(0)
        encoder_emb_input.append(c_agg_batch[0])
        encoder_type_input.append(0)
        for i, type_id in enumerate(neighbor_order_n_p_u[:npu]):
            if type_id[0] == 'p':
                encoder_emb_input.append(p_aft_rnn_dict[type_id[1]])
                encoder_type_input.append(1)
            elif type_id[0] == 'u':
                encoder_emb_input.append(u_aft_rnn_dict[type_id[1]])
                encoder_type_input.append(2)
            else:
                encoder_emb_input.append(n_aft_rnn_dict[type_id[1]])
                encoder_type_input.append(0)
            encoder_order_input.append(i + 1)
        encoder_type_input = torch.LongTensor(encoder_type_input).to(device)
        decoder_emb_input.append(c_agg_batch[0])
        decoder_type_input.append(0)
        for (i, type_id) in enumerate(news_neighbor):
            decoder_emb_input.append(n_aft_rnn_dict[type_id])
            decoder_order_input.append(i + 1)
            decoder_type_input.append(0)
        decoder_type_input = torch.LongTensor(decoder_type_input).to(device)
        encoder_emb_input = torch.stack(encoder_emb_input)
        self_att_c_all = self.mh_attention1(encoder_emb_input.view(bsz_all, -1, self.hidden_dim),
                                            encoder_emb_input.view(bsz_all, -1, self.hidden_dim), \
                                            encoder_emb_input.view(bsz_all, -1,self.hidden_dim))
        align_c_all = self.alignfc_c(self_att_c_all).view(bsz_all, self.hidden_dim)
        encoder_emb_input = align_c_all

        encoder_emb_input = encoder_emb_input.view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[1])

        decoder_emb_input = align_c

        decoder_emb_input = decoder_emb_input.view(decoder_emb_input.shape[0], 1, decoder_emb_input.shape[1])
        encoder_emb_input += self.type_encoder(encoder_type_input).view(encoder_emb_input.shape[0], 1,encoder_emb_input.shape[2])
        encoder_emb_input += self.pos_encoder(encoder_emb_input)
        encoder_emb_input += align_g_all.view(encoder_emb_input.shape[0], 1, encoder_emb_input.shape[2])

        decoder_emb_input += self.pos_encoder(decoder_emb_input)
        decoder_emb_input += self.type_encoder(decoder_type_input).view(decoder_emb_input.shape[0], 1,decoder_emb_input.shape[2])
        decoder_emb_input += align_g.view(decoder_emb_input.shape[0], 1,decoder_emb_input.shape[2])

        final_representation = self.transformer(encoder_emb_input, decoder_emb_input)
        return final_representation,dist

    def output(self, c_embed_batch):
        batch_size = 1
        c_embed = c_embed_batch[0, 0, :]
        c_embed = c_embed.view(batch_size, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        predictions = self.output_act(c_embed_out)
        return predictions

    def forward(self, x, neighbor_order_n_p_u,all_t):
        x,dist = self.transformer_agg(x, neighbor_order_n_p_u,all_t,npu=self.npu)
        x = self.output(c_embed_batch=x)
        return x,dist

def BCELoss(predictions, true_label):
    loss = nn.BCELoss()
    predictions = predictions.view(1)
    tensor_label = torch.FloatTensor(np.array([true_label])).to(device)
    loss_sum = loss(predictions, tensor_label)
    return loss_sum

def save_checkpoint(model, optimizer, save_path, epoch, val_acc):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, val_acc

def train_test(data_real, data_fake, test_size, PATH):
    y_real = range(len(data_real))
    y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size,
                                                                            random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size,
                                                                            random_state=42)
    np.savetxt(PATH +'train_index_real.txt', y_train_real)
    np.savetxt(PATH +'test_index_real.txt', y_test_real)
    np.savetxt(PATH +'train_index_fake.txt', y_train_fake)
    np.savetxt(PATH +'test_index_fake.txt', y_test_fake)

    return X_train_real, X_test_real, X_train_fake, X_test_fake

def fold_train_test(data_real, data_fake, test_size, PATH):
    y_real = range(len(data_real))
    y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size,
                                                                            random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size,
                                                                            random_state=42)
    np.savetxt(PATH +'train_index_real.txt', y_train_real)
    np.savetxt(PATH +'test_index_real.txt', y_test_real)
    np.savetxt(PATH +'train_index_fake.txt', y_train_fake)
    np.savetxt(PATH +'test_index_fake.txt', y_test_fake)

    # Shuffle the order in post nodes
    train_val = X_train_real + X_train_fake
    test_set = X_test_real + X_test_fake
    np.random.shuffle(train_val)
    np.random.shuffle(test_set)

    # K-fold validation index
    train_index = []
    val_index = []
    num_splits = 9
    kfold = KFold(num_splits, True, 1)
    for train, val in kfold.split(train_val):
        train_index.append(train)
        val_index.append(val)
    np.savetxt(PATH + 'fold_train_index.txt', train_index)
    np.savetxt(PATH + 'fold_val_index.txt', val_index)
    np.savetxt(PATH + 'fold_train_val.txt', train_val)
    np.savetxt(PATH + 'fold_test_set.txt', test_set)

    return train_index,val_index,train_val,test_set

def load_fold_train_test(PATH):
    train_index = np.loadtxt(PATH + 'fold_train_index.txt').astype('int32')
    train_index = train_index.tolist()
    val_index = np.loadtxt(PATH + 'fold_train_index.txt').astype('int32')
    val_index = val_index.tolist()
    train_val= np.loadtxt(PATH + 'fold_train_val.txt').astype('int32')
    train_val=train_val.tolist()
    test_set=np.loadtxt(PATH + 'fold_test_set.txt').astype('int32')
    test_set=test_set.tolist()

    return train_index,val_index,train_val,test_set

def load_train_test(data_real, data_fake, PATH):
    a = np.loadtxt(PATH+'test_index_real.txt')
    a = a.astype('int32')
    b = np.loadtxt(PATH+'test_index_fake.txt')
    b = b.astype('int32')

    test_set_real = []
    train_set_real = []
    for i in range(len(data_real)):
        if i in a:
            test_set_real.append(data_real[i])
        else:
            train_set_real.append(data_real[i])
    test_set_fake = []
    train_set_fake = []
    for j in range(len(data_fake)):
        if j in b:
            test_set_fake.append(data_fake[j])
        else:
            train_set_fake.append(data_fake[j])
    return train_set_real, test_set_real, train_set_fake, test_set_fake


# Initialize parameters
lr=0.001
num_epoch = 40
num_folds = 9
batch_size = 4
output_dim=208
hidden_dim=200
epsilon=1e-10
alp=2.5
belt=1.4

print('Start training')

five_res = []
base_PATH ='../pre-trained/GossipCop/'
if not os.path.exists(base_PATH):
    os.makedirs(base_PATH)
f_path = base_PATH + 'result.txt'

for repeat in range(5):
    PATH = base_PATH +'lr'+str(lr)+'bs'+str(batch_size)+ 'od'+str(output_dim)+'hd'+str(hidden_dim)+'al'+str(alp)+'be'+str(belt)+ '/repeat' + str(repeat) + '/'
    print("PATH:", PATH)
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    with open(f_path, "a") as f:
        f.write("lr:%s batch_size:%s\n" % (lr, batch_size))
        f.write("PATH:%s\n" % (PATH))
    print('Repeat:', repeat + 1)
    with open(f_path, "a") as f:
        f.write("Repeat:%s\n" % (repeat + 1))

    X_train_real, X_test_real, X_train_fake, X_test_fake= train_test(news_nodes_real, news_nodes_fake, 0.1, PATH)

    # Shuffle the order in post nodes
    train_val = X_train_real + X_train_fake
    test_set = X_test_real + X_test_fake
    np.random.shuffle(train_val)
    np.random.shuffle(test_set)

    # K-fold validation index
    train_index = []
    val_index = []
    num_splits = 9
    kfold = KFold(num_splits, True, 1)
    for train, val in kfold.split(train_val):
        train_index.append(train)
        val_index.append(val)

    for fold in range(num_folds):

        t0 = time.time()

        print("Start for fold", fold + 1)
        with open(f_path, "a") as f:
            f.write("Start for fold:%s\n" % (fold + 1))
        best_val = 0
        running_loss = 0.0
        val_loss = 0.0
        test_loss = 0.0
        best_epoch = 0
        start_epoch=0
        net = DPSG(input_dim=[768, 512, 3, 29],
                             n_hidden_dim=hidden_dim,
                             u_hidden_dim=hidden_dim,
                             p_hidden_dim=hidden_dim,
                             out_embed_d=output_dim, d_model=output_dim, attn_heads=8,enc_layers=1,  npu=30)
        net.to(device)
        net.init_weights()
        optimizer = optim.SGD(net.parameters(), lr=lr)

        cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
        print("save dir:",cur_PATH)
        if os.path.isfile(cur_PATH):
            with open(f_path, "a") as f:
                f.write("Read checkpoint")
            net, optimizer, start_epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)

        scheduler = StepLR(optimizer, step_size=4, gamma=0.01)
        t1 = time.time()
        print("init time: ", t1 - t0)
        with open(f_path, "a") as f:
            f.write("init time:%s \n" % (t1 - t0))
        for epoch in range(start_epoch,num_epoch):
            t_current_start = time.time()
            print('Epoch:', epoch + 1)
            with open(f_path, "a") as f:
                f.write("Epoch:%s\n" % (epoch + 1))
            m = 0.0
            train_loss = 0.0
            c = 0.0
            running_loss = 0.0
            running_loss_dis=0.0
            v = 0.0
            val_loss = 0.0
            val_loss_dis=0.0
            t = 0.0
            test_loss = 0.0
            real_count = 0.0
            fake_count = 0.0
            real_true = 0.0
            fake_true = 0.0
            train_set = []
            val_set = []
            for t_index in train_index[fold]:
                train_set.append(train_val[t_index])
            for v_index in val_index[fold]:
                val_set.append(train_val[v_index])
            random.shuffle(train_set)
            net.train()
            for i in range(len(train_set)):
                optimizer.zero_grad()
                all_t = []
                all_t.append(n_add_time_dict[train_set[i].node_id])
                all_t.extend(neighbor_dict[2][train_set[i].node_id])
                output,dist_og = net(train_set[i], neighbor_dict[0][train_set[i].node_id],all_t)
                out = output.item()
                if (out / (1 - out) >= ratio and train_set[i].label == 1) or (
                        out / (1 - out) < ratio and train_set[i].label == 0):
                    c += 1
                cur_loss = BCELoss(predictions=output, true_label=train_set[i].label)
                loss_mse = nn.MSELoss()
                cur_loss_dis = loss_mse(dist_og[0], dist_og[1])
                running_loss += cur_loss.item()
                running_loss_dis+=cur_loss_dis.item()
                if i % 100 == 99:
                    print('Repeat: %d,Fold: %d, Epoch: %d, step: %5d, loss: %.4f,loss_dis: %.4f,  acc: %.4f' %
                          (repeat + 1,fold + 1, epoch + 1, i + 1, running_loss / 100,running_loss_dis/ 100, c / 100))
                    with open(f_path, "a") as f:
                        f.write('Repeat: %d,Fold: %d, Epoch: %d, step: %5d, loss: %.4f,loss_dis: %.4f,  acc: %.4f \n' %
                                (repeat + 1,fold + 1, epoch + 1, i + 1, running_loss / 100,running_loss_dis/ 100, c / 100))
                    running_loss = 0.0
                    running_loss_dis = 0.0
                    c = 0.0
                if i % batch_size == 0:
                    loss = Variable(torch.zeros(1), requires_grad=True).to(device)
                    loss_dis= Variable(torch.zeros(1), requires_grad=True).to(device)
                loss = loss + cur_loss
                loss_dis=loss_dis+cur_loss_dis
                if i % batch_size == (batch_size - 1):
                    loss_defense=alp*loss+belt*loss_dis
                    loss_defense = loss_defense / batch_size
                    loss_defense.backward()
                    optimizer.step()
            net.eval()
            for j in range(len(val_set)):
                all_t = []
                all_t.append(n_add_time_dict[train_set[i].node_id])
                all_t.extend(neighbor_dict[2][train_set[i].node_id])
                output,dist_og= net(val_set[j], neighbor_dict[0][val_set[j].node_id],all_t)
                out = output.item()
                if val_set[j].label == 1:
                    real_count += 1
                    if out / (1 - out) >= ratio:
                        real_true += 1
                else:
                    fake_count += 1
                    if out / (1 - out) < ratio:
                        fake_true += 1
                if (out / (1 - out) >= ratio and val_set[j].label == 1) or (
                        out / (1 - out) < ratio and val_set[j].label == 0):
                    v += 1
                vloss = BCELoss(predictions=output, true_label=val_set[j].label)
                loss_mse = nn.MSELoss()
                vloss_dis = loss_mse(dist_og[0], dist_og[1])
                val_loss += vloss.item()
                val_loss_dis += vloss_dis.item()
            val_acc = v / len(val_set)
            real_precision = real_true / (real_true + fake_count - fake_true)
            fake_precision = fake_true / (fake_true + real_count - real_true)
            real_recall = real_true / real_count
            fake_recall = fake_true / fake_count
            real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
            fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
            print('Validation loss: %.4f, Validation loss_dis: %.4f,Validation accuracy: %.4f' % (val_loss / len(val_set), val_loss_dis / len(val_set),val_acc))
            print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
            print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))
            with open(f_path, "a") as f:
                f.write('Validation loss: %.4f, Validation loss_dis: %.4f,Validation accuracy: %.4f\n' % (val_loss / len(val_set), val_loss_dis / len(val_set),val_acc))
                f.write(
                    'Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f\n' % (real_precision, real_recall, real_f1))
                f.write(
                    'Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f\n' % (fake_precision, fake_recall, fake_f1))
            if val_acc >= best_val:
                print('Update model at epoch:', epoch + 1)
                with open(f_path, "a") as f:
                    f.write('Update model at epoch:%s\n' % (epoch + 1))
                cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
                save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
                best_val = val_acc
                best_epoch = epoch
            scheduler.step()
            if epoch - best_epoch >= 5:
                break
            t_current_end = time.time()
            print("time for a epoch: ", t_current_end - t_current_start, )  # "num of neighbor: ", npu)
            with open(f_path, "a") as f:
                f.write('time for a epoch:%s\n' % (t_current_end - t_current_start))
    print('Finish training')

    print('==============================================================')
    with open(f_path, "a") as f:
        f.write('Finish training')


    print("Init net and optimizer skeletons")
    with open(f_path, "a") as f:
        f.write("\nInit net and optimizer skeletons\n")

    # Init net and optimizer skeletons
    net = DPSG(input_dim=[768, 512, 3, 29],
                         n_hidden_dim=hidden_dim, u_hidden_dim=hidden_dim, p_hidden_dim=hidden_dim,
                         out_embed_d=output_dim, d_model=output_dim, attn_heads=8,
                         enc_layers=1, npu=30)
    net.to(device)
    net.init_weights()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    print("load all folds best_model")
    with open(f_path, "a") as f:
        f.write("load all folds best_model\n")
    best_models = []
    for fold in range(num_folds):
        cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
        net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
        print("fold:%s best_val:%s" % (fold, best_val))
        with open(f_path, "a") as f:
            f.write("fold:%s best_val:%s\n" % (fold, best_val))
        net.eval()
        best_models.append(net)

    t = 0.0
    test_loss = 0.0
    test_loss_dis = 0.0
    real_count = 0.0
    fake_count = 0.0
    real_true = 0.0
    fake_true = 0.0
    print("Test Starting")
    with open(f_path, "a") as f:
        f.write("\nTest Starting\n")
    for k in range(len(test_set)):
        output = 0.0
        avg_tloss = 0.0
        avg_tloss_dis = 0.0
        for fold in range(num_folds):
            all_t = []
            all_t.append(n_add_time_dict[train_set[i].node_id])
            all_t.extend(neighbor_dict[2][train_set[i].node_id])
            result,dist_og = best_models[fold](test_set[k], neighbor_dict[0][test_set[k].node_id],all_t)
            output += result.item()
            tloss = BCELoss(predictions=result, true_label=test_set[k].label)
            loss_mse = nn.MSELoss()
            tloss_dis = loss_mse(dist_og[0], dist_og[1])
            avg_tloss += tloss.item()
            avg_tloss_dis += tloss_dis.item()
        output /= num_folds
        avg_tloss /= num_folds
        avg_tloss_dis /= num_folds
        test_loss += avg_tloss
        test_loss_dis += avg_tloss_dis
        if (output / (1 - output) >= ratio and test_set[k].label == 1) or (
                output / (1 - output) < ratio and test_set[k].label == 0):
            t += 1
        if test_set[k].label == 1:
            real_count += 1
            if output / (1 - output) >= ratio:
                real_true += 1
        else:
            fake_count += 1
            if output / (1 - output) < ratio:
                fake_true += 1

    real_precision = real_true / (real_true + fake_count - fake_true)
    fake_precision = fake_true / (fake_true + real_count - real_true)
    real_recall = real_true / real_count
    fake_recall = fake_true / fake_count
    real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
    fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
    print('Test loss: %.4f, Test loss_dis: %.4f, Test accuracy: %.4f' % (test_loss / len(test_set), test_loss_dis / len(test_set),t / len(test_set)))
    print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
    print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))
    with open(f_path, "a") as f:
        f.write('Test loss: %.4f, Test loss_dis: %.4f, Test accuracy: %.4f\n' % (test_loss / len(test_set), test_loss_dis / len(test_set),t / len(test_set)))
        f.write('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f\n' % (real_precision, real_recall, real_f1))
        f.write('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f\n' % (fake_precision, fake_recall, fake_f1))
    res = []
    res.append(round((test_loss / len(test_set)),4))
    res.append(round((test_loss_dis / len(test_set)), 4))
    res.append(round(t / len(test_set), 4))
    res.append(round(real_precision, 4))
    res.append(round(real_recall, 4))
    res.append(round(real_f1, 4))
    res.append(round(fake_precision, 4))
    res.append(round(fake_recall, 4))
    res.append(round(fake_f1, 4))
    five_res.append(res)
with open(f_path, "a") as f:
    for i in five_res:
        f.write((' '.join(map(str, i))) + '\n')
for i in five_res:
    print(i)



