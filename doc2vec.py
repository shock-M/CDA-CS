import sys  # doc2vev
import gensim
import sklearn
import numpy as np

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from sklearn.feature_extraction.text import TfidfVectorizer

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_datasest():
    with open("python_code.original_subtoken", 'r', encoding='utf-8') as cf:
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


def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)


def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, vector_size=5, window=2, min_count=1, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save("model/python_code.model")

    return model_dm


def test():
    model_dm = Doc2Vec.load("model/exp.model")
    print(model_dm)
    test_text = ['i', 'am', 'happy']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)

    return sims


if __name__ == '__main__':
    x_train = get_datasest()
    model_dm = train(x_train)

    # sims = test()
    # for count, sim in sims:
    #     sentence = x_train[count]
    #     words = ''
    #     for word in sentence[0]:
    #         words = words + word + ' '
    #     print(words, sim, len(sentence[0]))
    # print('ok')
    model_dm1 = Doc2Vec.load("model/python_code.model")
    model_dm2 = Doc2Vec.load("model/python_nl.model")

    with open("python_code_nl_features.txt", 'w') as f:
        for index in range(55537):
            # print(model_dm1.docvecs[69707]+model_dm2.docvecs[69707])
            c2nl = model_dm1.docvecs[index]+model_dm2.docvecs[index]
            c2nl = model_dm1.docvecs[index]
            temp = ""
            for i in range(len(c2nl)):
                if i <= len(c2nl)-1:
                    temp += str(c2nl[i]) + '\t'
                else:
                    temp += str(c2nl[i]) + '\t'
            f.writelines(temp + '\n')
            # print(temp)

