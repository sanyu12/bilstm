import gensim
import numpy as np

def retrain(words, model):
    pass

def WordsVect(words, model_vectors, unkown_words=None):
    """
    :param words: list
    :return: vect : 64 D
    """
    vect = list()
    un_wors = dict()
    for i in words:
        try:
            c = model_vectors[i]
            vect.append(c.tolist())
        except KeyError:
            if unkown_words and i in unkown_words.keys():
                c = unkown_words[i]
            else:
                # print(i, " is not in voc.")
                c = np.random.normal(0, 10, 256)
                c = (c - c.mean()) / c.std()
                c = (c - c.min()) / (c.max() - c.min())
                c = c.tolist()
                un_wors[i] = c
            vect.append(c)
    return vect, un_wors

def TagsOneHot(Tags, tag_size):
    tmp = np.zeros(shape=(len(Tags), tag_size))
    for i in range(len(Tags)):
        tmp[i, Tags[i]] = 1.0
    return tmp.tolist()

# testing 64
# vect_model = gensim.models.KeyedVectors.load_word2vec_format('D:/project/bilstm/model/'
#                                                              'news_12g_baidubaike_20g_novel_90g_embedding_64.bin', binary=True)
# print(WordsVect(["你好","微信"], vect_model))
# tags = [1,2,3,4,5]
# # print(TagsOneHot(tags, 10))
# model = gensim.models.Word2Vec.load('D:/project/bilstm/model/word2vec_from_weixin/word2vec/word2vec_wx')
# print(model[u'探'])
# 一九九八年