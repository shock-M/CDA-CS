# coding=utf-8
import copy

from eda_uda_bert import *
from tfidf import TfIdfWordRep, compute_tfidf
import random
#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

def Kmeans():
    # -*- coding: utf-8 -*-
    import pandas as pd
    from sklearn import metrics
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    # 参数初始化
    inputfile = 'tfidf_feature/java/java_nl_features.xlsx'
    outputfile = 'java_nl.xlsx'
    k = 8  # 聚类的类别
    iteration = 500  # 聚类最大循环次数
    # data = pd.read_excel(inputfile, index_col = 'Id') #读取数据
    data = pd.read_excel(inputfile)  # 读取数据
    data_zs = 1.0 * (data - data.mean()) / data.std()  # 数据标准化

    # #SSE
    # d=[]
    # for i in range(1,20):    #k取值1~15，做kmeans聚类，看不同k值簇内误差平方和
    #     km=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    #     km.fit(data_zs)
    #     d.append(km.inertia_)  #inertia簇内误差平方和
    #     print(i)
    #
    # plt.plot(range(1, 20), d, marker='o')
    # plt.xlabel('number of clusters')
    # plt.ylabel('distortions')
    # plt.show()

    # # 轮廓系数
    # from sklearn import preprocessing
    # #正则化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_norm = min_max_scaler.fit_transform(data_zs)
    # scores = []
    # for i in range(2, 15):
    #     km = KMeans(        n_clusters=i,
    #                         init='k-means++',
    #                         n_init=10,
    #                         max_iter=300,
    #                         random_state=0      )
    #     km.fit(X_norm)
    #     scores.append(metrics.silhouette_score(X_norm, km.labels_, metric='euclidean'))
    # plt.plot(range(2, 15), scores, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('silhouette_score')
    # plt.show()

    model = KMeans(n_clusters=k, max_iter=iteration)  # 分为k类
    model.fit(data_zs)  # 开始聚类

    # 简单打印结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2, r1], axis=1)  # 横向连接(0是纵向), 得到聚类中心对应的类别下的数目
    r.columns = list(data.columns) + [u'类别数目']  # 重命名表头
    print(r)
    # 详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)  # 详细
    # 输出每个样本对应的类别
    r.columns = list(data.columns) + [u'聚类类别']  # 重命名表头
    r.to_excel(outputfile)  # 保存结果

    res0Series = pd.Series(model.labels_)
    ## num 每一类的数量
    num = []
    for i in range(k):
        res0 = res0Series[res0Series.values == i]
        num.append(len(res0))

    # 求m个最小的数值及其索引
    t = copy.deepcopy(num)
    min_number = []
    min_index = []
    for _ in range(6):
        number = min(t)
        index = t.index(number)
        t[index] = 1000000
        min_number.append(number)
        min_index.append(index)
    t = []
    print(min_number)
    print(min_index)
    a = min_index[0]
    b = min_index[1]
    c = min_index[2]
    d = min_index[3]
    e = min_index[4]
    f = min_index[5]

    resa = res0Series[res0Series.values == a]
    resb = res0Series[res0Series.values == b]
    resc = res0Series[res0Series.values == c]
    resd = res0Series[res0Series.values == d]
    rese = res0Series[res0Series.values == e]
    resf = res0Series[res0Series.values == f]
    res = resa + resb + resc + resd + rese + resf
    # res = sum(resa, resb)
    # res = sum(res, resc)
    # res = sum(res, resd)
    # res = sum(res, rese)
    ## 最少的几类

    # if(res0Series.values == a or res0Series.values == b or res0Series.values == c):
    #     res = res0Series[res0Series.values ]

    # if(res0Series.values == min_index[0]):
    #     res += res0Series[res0Series.values]
    # if(res0Series.values == min_index[1]):
    #     res += res0Series[res0Series.values]
    # if (res0Series.values == min_index[2]):
    #     res += res0Series[res0Series.values]

    #res = res0Series[res0Series.values in min_index]
    # res = sum(res, [])
    print(res)
    print("index")
    #print(res.index)
    return res.index

    # ## 最少的一类
    # min_index = num.index(min(num))
    # min_res = res0Series[res0Series.values == min_index]
    # print(min_res)
    # print(min_res.index)
    # return min_res.index
    # lines = open("data/java/nl/java_nl.original", 'r', encoding='utf-8').readlines()
    #
    # for i, line in enumerate(lines):
    #     if i in min_res.index:
    #         print("开始增强")

def rand_copy(label, sentence):
    aug_sentences = []
    for _ in range(5):
        aug_sentences.append(sentence)
    return aug_sentences


#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()

    ## tf-idf
    sents = []
    tf_idf = {}
    idf = {}
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        sents.append(parts[1])
    tf_idf, idf = compute_tfidf(sents)
    print("finish compute tf_idf...........")
    token_prob = 0.5
    tfidf = TfIdfWordRep(token_prob, tf_idf, idf)
    min_index = Kmeans()
    # sum = 0
    # ########## Kmeans #############
    # for i, line in enumerate(lines):
    #     parts = line[:-1].split('\t')
    #     label = parts[0]
    #     sentence = parts[1]
    #     if i in min_index:
    #         # rand = random.random()
    #         # if rand < 1:
    #         #     print(rand)
    #         aug_sentences = rand_copy(label, sentence)
    #         for aug_sentence in aug_sentences:
    #             sum += 1
    #             writer.write(label + "\t" + aug_sentence + '\n')
    #     else:
    #         writer.write(label + "\t" + sentence + '\n')
    # print(sum)
    ########## Kmeans + eda #########
    for i, line in enumerate(lines):
        if i in min_index:
            #print("line: "+line)
            parts = line[:-1].split('\t')
            #print("parts: "+parts[1]+'\n')
            label = parts[0]
            sentence = parts[1]
            #print("parts: " + label )
            aug_sentences = eda(tfidf, sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
            for aug_sentence in aug_sentences:
                writer.write(label + "\t" + aug_sentence + '\n')
    # ######### eda #######
    # for i, line in enumerate(lines):
    #     #print("line: "+line)
    #     parts = line[:-1].split('\t')
    #     #print("parts: "+parts[1]+'\n')
    #     label = parts[0]
    #     sentence = parts[1]
    #     #print("parts: " + label )
    #     aug_sentences = eda(tfidf, sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
    #     for aug_sentence in aug_sentences:
    #         writer.write(label + "\t" + aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
