from operator import index
import numpy as np
import pandas as pd
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from tqdm import tqdm


def check_and_download_nltk_data(package_name):
    try:
        nltk.data.find(f"corpora/{package_name}")
        print(f"'{package_name}' package is already downloaded.")
    except LookupError:
        print(f"'{package_name}' package not found, downloading...")
        nltk.download(package_name)


def cut_words(filename):
    word_list = []
    porter = PorterStemmer()
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if not line.strip():
                continue
            tokens = word_tokenize(line)
            tokens = [token.lower() for token in tokens]
            # remove signs

            stripped_tokens = [token.translate(table) for token in tokens if token.isalpha()]

            words = [word for word in stripped_tokens if word not in stop_words]
            # stemming
            stemmed = [porter.stem(word) for word in words]
            word_list.append(stemmed)
    return word_list

def get_dict(data):
    index_2_word = []
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)
    word_2_index = {word: index for index, word in enumerate(index_2_word)}
    word_size = len(word_2_index)
    word_2_onehot = {}
    for word, index in word_2_index.items():
        one_hot = np.zeros((1, word_size))
        one_hot[0, index] = 1
        word_2_onehot[word] = one_hot
    return index_2_word, word_2_index, word_2_onehot


def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex, axis=1, keepdims = True)


# check_and_download_nltk_data('punkt')
# check_and_download_nltk_data('stopwords')
word_list = cut_words('indep.txt')
index_2_word, word_2_index, word_2_onehot = get_dict(word_list)
word_size = len(word_2_index)
embedding_num = 107 #100 - 300
lr = 0.01 # learning rate
epoch = 10 # training rounds
n_gram = 3 # neighbor elements number, like sliding window
w1 = np.random.normal(-1,1, size = (word_size, embedding_num))
w2 = np.random.normal(-1,1, size = (embedding_num, word_size))
for i in range(epoch):
    for words in tqdm(word_list):
        for idx, word in enumerate(words):
            # edge case
            cur_one_hot = word_2_onehot[word]
            surronding_words = words[max(idx-n_gram, 0) : idx] + words[idx+1 : idx+1+n_gram]
            for surronding_word in surronding_words:
                surronding_word_onehot = word_2_onehot[surronding_word]
                hidden = cur_one_hot @ w1
                p = hidden @ w2
                pre = softmax(p)
                
                # loss = -np.sum(surronding_word_onehot * np.log(pre))
                # A @ B = C
                # dC = G
                # dA = G @ BT
                # dB = AT @ G
                G2 = pre - surronding_word_onehot
                delta_w2 = hidden.T @ G2
                G1 = G2 @ w2.T
                delta_w1 = cur_one_hot.T @ G1

                w1 -= lr * delta_w1
                w2 -= lr * delta_w2

with open('word2vec.pkl', 'wb') as f:
    pickle.dump([w1, word_2_index, index_2_word, w2],f) # word2vec 负采样