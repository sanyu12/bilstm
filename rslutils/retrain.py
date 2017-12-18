import gensim
import logging
from rslutils import prepocess
def build_sentences(data):
    sentence = list()
    for index in range(len(data["Data"])):
        words, _, _ = prepocess.WordSplit(data["Data"][index], train=False)
        if len(words) == 0:
            continue
        sentence.append(words)
    return  sentence

train_data = prepocess.ReadFunc(u"D:/project/bilstm/data/cpbtrain.txt")
val_data = prepocess.ReadFunc(u"D:/project/bilstm/data/cpbdev.txt")
test_data = prepocess.ReadFunc(u"D:/project/bilstm/data/cpbtest.txt")

sentences = list()
sentences += build_sentences(train_data)
sentences += build_sentences(val_data)
sentences += build_sentences(test_data)


logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO)

model = gensim.models.Word2Vec.load('D:/project/bilstm/model/word2vec_from_weixin/word2vec/word2vec_wx')

model.build_vocab(sentences, keep_raw_vocab=True, update=True)
for i in range(2):
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter, compute_loss=False)
    print(model.most_similar(u'查瑞传'))

model.save('D:/project/bilstm/model/word2vec_from_weixin/word2vec_wx')