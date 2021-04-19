import numpy as np
import pandas as pd
import gensim
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    with open("java/java_nl.original", 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
        print(len(docs))

    x_train = []
    # y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        # 创建对象
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train

def tfidf_vec(lines):

    x_train = []
    for i, line in enumerate(lines):
        x_train.append(line)

    # x_train = get_datasest()

    print(x_train[0])
    all_x = x_train  # 原始数据集
    tfv = TfidfVectorizer(min_df=3, max_features=None,
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words='english')
    tfv.fit(all_x)
    all_x = tfv.transform(all_x)
    # train_len = len(t_set)
    # x_train = all_x[:train_len]

    # x_test = all_x[train_len:]
    # y_train = t_set_df['sentiment']

    # 奇异值分解，将词向量降到100维
    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    x_train = lsa.fit_transform(all_x)  # 训练集大小
    # x_test = lsa.fit_transform(x_test)  # 测试集大小
    return x_train

def file(path, vec):
    with open(path, 'w') as f:
        for index in range(69708):
            c2nl = vec[index]
            temp = ""
            for i in range(len(c2nl)):
                if i <= len(c2nl) - 1:
                    temp += str(c2nl[i]) + '\t'
                else:
                    temp += str(c2nl[i]) + '\t'
            f.writelines(temp + '\n')
            # print(temp)
        print(vec[55538])

def file2(path, code, nl):
    with open(path, 'w') as f:
        for index in range(69708):
            c2nl = code[index] + nl[index]
            temp = ""
            for i in range(len(c2nl)):
                if i <= len(c2nl) - 1:
                    temp += str(c2nl[i]) + '\t'
                else:
                    temp += str(c2nl[i]) + '\t'
            f.writelines(temp + '\n')
            # print(temp)
        print(code[55538])
        print(nl[55538])
        print(code[55538] + nl[55538])

lines_code = open("java/java_code.original_subtoken", 'r', encoding='utf-8').readlines()
code = tfidf_vec(lines_code)
lines_nl = open("java/java_nl.original", 'r', encoding='utf-8').readlines()
nl = tfidf_vec(lines_nl)

file("java/java_code_features.txt", code)
file("java/java_nl_features.txt", nl)
file2("java/java_code_nl_features.txt", code, nl)




