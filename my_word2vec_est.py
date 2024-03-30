import pickle
import numpy as np
w1, word_2_index, index_2_word, w2 = pickle.load(open('word2vec.pkl', 'rb'))

def word_voc(word):
    return w1[word_2_index[word]]

def voc_sim(word, top_n):
    v_w1 = word_voc(word)
    word_sim = {}
    for i in range(len(word_2_index)):
        v_w2 = w1[i]
        theta_sum = np.dot(v_w1, v_w2)
        theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
        theta = theta_sum/theta_den
        word = index_2_word[i]
        word_sim[word] = theta
    words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
    for word, sim in words_sorted[:top_n]:
        print(word, sim)

voc_sim('hola', 20)
