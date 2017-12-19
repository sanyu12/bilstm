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

train_data = prepocess.ReadFunc(u"./data/cpbtrain.txt")
val_data = prepocess.ReadFunc(u"./data/cpbdev.txt")
test_data = prepocess.ReadFunc("./data/cpbtest.txt")

t_dic, t_idx, l_dic, l_idx = calc_idx([train_data, val_data, test_data])
print(l_dic)

srl_model = SRLModel('./model/word2vec_wx',
                 labels=l_dic, embedding_size=256, hidden_layer=100, nlabels=len(l_idx),
                 tag_size=len(t_idx), pad_tok=0)

# test data loader
test_seq, test_tags, _, test_rel = prepocess.build_data(test_data, test=True)
test_tag_int = get_idx(test_tags, t_dic)

srl_model.restoreModel("tmp_2_model_2000")
#
viterbi_sequences, seq_lengths = srl_model.predict(test_seq, test_tag_int, test_rel, 100)

# filename, dev_file, viterbi_seq, seq_length, val_data, val_tags
srl_model.evaluate("pred_test", "null", viterbi_sequences, seq_lengths, test_data, test_tags)


# with open("res", "wb") as f:
#     pickle.dump(viterbi_sequences, f)
#1115 1115
# recall: 0.649407114624506  precision: 0.7481785063752276  F: 0.6953025814642403
#1115 1115
# recall: 0.6470355731225297  precision: 0.7689055894786284  F: 0.7027259068469629
#1115 1115 200
# recall: 0.7201581027667984  precision: 0.735270379338176  F: 0.7276357827476038
#
# with open("len_res", "wb") as f:
#     pickle.dump(seq_lengths, f)

print("end")