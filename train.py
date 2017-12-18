import collections
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
s, t, l, rel = prepocess.build_data(train_data)

srl_model = SRLModel('D:/project/bilstm/model/word2vec_from_weixin/word2vec/word2vec_wx',
                 labels=l_dic, embedding_size=256, hidden_layer=50, nlabels=len(l_idx),
                 tag_size=len(t_idx), pad_tok=0)

train_tags = get_idx(t, t_dic)
train_labels = get_idx(l, l_dic)

# test data loader
val_seq, val_tags, val_label, val_rel = prepocess.build_data(val_data)
val_tag_int = get_idx(val_tags, t_dic)
val_label_int = get_idx(val_label, l_dic)
srl_model.load_test(val_seq, val_tag_int, val_rel, val_label_int, val_tags)

# train
srl_model.train(s, train_tags, train_labels, rel, 10000, 100,
                'D:/project/bilstm/cpbdev.txt')
# save
srl_model.save_session("srlmodel")

srl_model.close_session()
