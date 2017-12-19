import numpy as np
import pandas as pd
import sys
import collections

def ReadFunc(file_name):
    Data = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            Data.append(line[:-1])
    return pd.DataFrame(Data, columns=['Data'])

def printList(args_list, sp='\n'):
    for i in args_list:
        sys.stdout.write("%s%s" % (i, sp))

def idx_tags(tags):
    tmp = list()
    for i in tags:
        tmp.extend(i)
    t_dict = collections.Counter(tmp)
    index_of_labels = t_dict.keys()
    list_of_labels = list()
    dict_of_labels = dict()
    num = 0
    for label in index_of_labels:
        dict_of_labels[label] = num
        list_of_labels.append(label)
        num += 1
    return dict_of_labels, list_of_labels

def WordSplit(Str, train=True):
    tmp = []
    wordAttr = []
    labels = []
    rel_loc = []
    if Str == "":
        return tmp, wordAttr, labels, rel_loc
    for i in Str[:-1].split(' '):
        sp = i.split('/')
        tmp.append("".join(sp[0]))
        wordAttr.append(sp[1])
        if train:
            if sp[2]== 'rel':
                rel_loc.append(sp[0])
            labels.append(sp[2])
        else:
            if len(sp) > 2:
                rel_loc.append(sp[0])
    return tmp, wordAttr, labels, rel_loc

def build_data(data, test=False):
    sentence = list()
    label = list()
    tag = None
    rel_locs = list()
    if not test:
        tag = list()
    for index in range(len(data["Data"])):
        if data["Data"][index] == "":
            continue
        if not test:
            words, labels, tags, rel_loc = WordSplit(data["Data"][index], train=True)
        else:
            words, labels, tags, rel_loc = WordSplit(data["Data"][index], train=False)
        if len(words) == 0:
            continue
        sentence.append(words)
        label.append(labels)
        if not test:
            tag.append(tags)
        rel_locs.append(rel_loc)
    return  sentence, label, tag, rel_locs

    # B E
def check_validity(pred_file):
    preds = [pred.split() for pred in open(pred_file, 'r', encoding="utf-8").read().strip().split('\n')]
    num = 0
    for pred in preds:
        num += 1
        lastname = ''
        keys_pred = dict()
        for item in pred:
            word, label = item.split('/')[0], item.split('/')[-1]
            flag, name = label[:label.find('-')], label[label.find('-') + 1:]
            if flag == 'O':
                continue
            if flag == 'S':
                if name not in keys_pred:
                    keys_pred[name] = [word]
                else:
                    keys_pred[name].append(word)
            else:
                if flag == 'B':
                    if name not in keys_pred:
                        keys_pred[name] = [word]
                    else:
                        keys_pred[name].append(word)
                    lastname = name
                elif flag == 'I' or flag == 'E':
                    if name == lastname : # "the I-/E- labels are inconsistent with B- labels in pred file."
                        keys_pred[name][-1] += ' ' + word
                    else:
                        print("ids: ", num, " ",word)


