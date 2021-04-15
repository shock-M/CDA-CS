#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

import numpy as np
import math
# import tensorflow as tf

# from augmentation import aug_policy
# import word_level_augment

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize


def compute_tfidf(sents):
    """

    print("sents")
    print(sents)
    corpus = TextCollection(sents)
    print("corpus")
    #print(corpus)
    tf_idf = {}
    idf = {}
    for i in range(len(corpus)):
        tf_idf[corpus[i]] = corpus.tf_idf(corpus[i], corpus)
        #print(corpus[i])
        idf[corpus[i]] = corpus.idf(corpus[i])
    print("#######")
    #print(tf_idf)
    return tf_idf, idf
    """
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    examples = [word_tokenize(sent) for sent in sents]  # examples是每一句将单词分开后的
    print("sents")
    print(len(sents))
    #print(sents)
    print("examples")
    #print(examples)
    word_doc_freq = collections.defaultdict(int)

    # Compute IDF
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i])
        for word in cur_sent:
            # if word == '"':
            #    print('cur_sent')
            #    print(cur_sent)
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i])
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return idf, tf_idf



class TfIdfWordRep():
    def __init__(self, token_prob, tf_idf, idf):
        self.token_prob = token_prob
        self.tf_idf = tf_idf
        self.idf = idf
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        # all_words = word_tokenize(sent)
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            # if word == '"':
            #    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #   print(all_words)
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        if replace_prob.size != 0:
            replace_prob = np.max(replace_prob) - replace_prob
            replace_prob = (replace_prob / replace_prob.sum() *
                            self.token_prob * len(all_words))
        return replace_prob

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
          self.reset_random_prob()
        return value


