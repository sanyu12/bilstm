import collections
import numpy as np
import pickle
from rslutils import prepocess
from tmodel.model import SRLModel

def generate(seq):
    dic, idx = prepocess.idx_tags(seq)
    return dic, idx

def calc_idx(data):
    labels, tags = list(), list()
    for it in data[:-1]:
        _, t, l, _ = prepocess.build_data(it)
        tags += t
        labels += l
    _, t, l, _ = prepocess.build_data(data[-1], test=True)
    tags += t

    t_dic, t_idx = prepocess.idx_tags(tags)
    l_dic, l_idx = prepocess.idx_tags(labels)
    return t_dic, t_idx, l_dic, l_idx

def get_idx(data, data_dict):
    id = list()
    for ele in data:
        seq_id = []
        for i in range(len(ele)):
            seq_id.append(data_dict[ele[i]])
        id.append(seq_id)
    return id

import sys
if __name__ == '__main__':

    if len(sys.argv[1:]) != 2:
        print('the function takes exactly two parameters: pred_file and gold_file')

    predict_model_name = sys.argv[1]
    save_predict_name = sys.argv[2]

    train_data = prepocess.ReadFunc(u"./data/cpbtrain.txt")
    val_data = prepocess.ReadFunc(u"./data/cpbdev.txt")
    test_data = prepocess.ReadFunc("./data/cpbtest.txt")

    t_dic, t_idx, l_dic, l_idx = calc_idx([train_data, val_data, test_data])

    srl_model = SRLModel('./model/word2vec_wx',
                         labels=l_dic, embedding_size=256, hidden_layer=200, nlabels=len(l_idx),
                         tag_size=len(t_idx), pad_tok=0)

    test_seq, test_tags, _, test_rel = prepocess.build_data(test_data, test=True)
    test_tag_int = get_idx(test_tags, t_dic)

    srl_model.restoreModel(predict_model_name, 100)

    viterbi_sequences, seq_lengths = srl_model.predict(test_seq, test_tag_int, test_rel, test_tags, 100)

    srl_model.evaluate(save_predict_name, "null", viterbi_sequences, seq_lengths, test_seq, test_tags)

    print("End")