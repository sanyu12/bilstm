import tensorflow as tf
import numpy as np
import gensim
import os
import pickle
from rslutils.embeding import WordsVect, TagsOneHot
from rslutils.calc_f1 import calc_f1
class SRLModel():
    def __init__(self, model_file, labels, embedding_size, hidden_layer, nlabels, tag_size, pad_tok):
        # labels, all label impossible
        self.label_to_tag = {idx: tag for tag, idx in labels.items()}
        self.embedding_size = embedding_size
        self.hidden_layer = hidden_layer
        self.nlabels = nlabels
        self.tag_size = tag_size
        self.pad_tok = pad_tok
        self._cursor = 0
        self.data_size = 0
        self.labels = 0
        self.test_data = []
        self.test_raw_data = []
        self.test_raw_tags = []
        self.test_label = 0
        self.dropout = 0
        self.unknown_words = dict()
        self.restore = 0
        self.learn_rate = 0.002
        self.alg_optim = "adam"
        self.clip_val = 1.25
        # 'D:/project/bilstm/model/word2vec_from_weixin/word2vec_wx'
        model = gensim.models.Word2Vec.load(model_file)
        self.model_vectors = model.wv
        del model


    def next_batch(self, batch_size, data, tags, rel_loc, labels=None ):
        # data, tags, labels, sequences of words, tag or labels are int
        batches = []
        rlabels = []

        if self._cursor == 0:
            self.data_size = len(data)
            segment = self.data_size // batch_size
            self._cursor = [offset * segment for offset in range(batch_size)]

        for b in range(batch_size):
            wv, un = WordsVect(data[self._cursor[b]], self.model_vectors, self.unknown_words)
            self.unknown_words = dict(self.unknown_words, **un)
            tgs = TagsOneHot(tags[self._cursor[b]], self.tag_size)
            st, un = WordsVect(rel_loc[self._cursor[b]], self.model_vectors, self.unknown_words) #[0] * len(wv)
            self.unknown_words = dict(self.unknown_words, **un)
            st = st * len(wv)
            arr = np.concatenate((np.array(wv) , np.array(st), np.array(tgs)),axis=1)
            batches += [arr.tolist()]
            if labels:
                rlabels.append(labels[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self.data_size

        return batches, rlabels


    def load_test(self, test, tags, rel_loc, labels=None, raw_tags=None):
        # data, tags, labels, sequences of words, tag or labels are int
        batches = []
        rlabels = []
        if len(self.test_data) == 0:
            for b in range(len(test)):
                wv, un = WordsVect(test[b], self.model_vectors, self.unknown_words)
                self.unknown_words = dict(self.unknown_words, **un)
                tgs = TagsOneHot(tags[b], self.tag_size)
                st, un = WordsVect(rel_loc[b], self.model_vectors, self.unknown_words)  # [0] * len(wv)
                self.unknown_words = dict(self.unknown_words, **un)
                st = st * len(wv)
                arr = np.concatenate((np.array(wv), np.array(st), np.array(tgs)), axis=1)
                if labels:
                    rlabels.append(labels[b])
                batches += [arr.tolist()]
            self.test_data = batches
            self.test_label = rlabels
            self.test_raw_data = test
            self.test_raw_tags = raw_tags
            return batches, rlabels
        else:
            return self.test_data, self.test_label


    def _seq_padding(self, pad_tok, data, max_length, nlevels=1):

        sequence_padded, sequence_length = [], []
        if nlevels == 1:
            for seq in data:
                seq = list(seq)
                seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
                sequence_padded += [seq_]
                sequence_length += [min(len(seq), max_length)]
        else:
            length_list = len(data[0][0])
            for seq in data:
                seq = list(seq)
                seq_ = seq[:max_length] + [ [pad_tok] * length_list ] * max(max_length - len(seq), 0)
                sequence_padded += [seq_]
                sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length


    def get_feed_dict(self, pad_tok, batch_seq, labels=None):
        feed = None
        max_length = max(map(lambda x: len(x), batch_seq))
        if labels:
            label_seq, _ = self._seq_padding(pad_tok, labels, max_length)
            feed = {
                self.labels: label_seq,
                self.dropout: 0.5
            }
        else:
            feed = {
                self.dropout: 1.0
            }

        sequence_padded, sequence_length = self._seq_padding(pad_tok, batch_seq, max_length, nlevels=2)
        feed[self.sequence_lengths] = sequence_length
        feed[self.word_data] = sequence_padded
        return feed, sequence_length


    def _build_tf(self, batch_size, lr_method, lr, clip):
        # shape = (batch size, max length of sentence in batch)
        self.word_data = tf.placeholder(tf.float32, shape=[None, None, 256 * 2 + self.tag_size],
                                        name="word_data")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                                     name="labels")
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")

        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_layer)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_layer)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.word_data, sequence_length=self.sequence_lengths, dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # TODO: dropout
        output = tf.nn.dropout(output, self.dropout)

        # crf-loss
        # TODO: Logits
        W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.hidden_layer, self.nlabels])
        b = tf.get_variable("b", shape=[self.nlabels],
                            dtype=tf.float32, initializer=tf.zeros_initializer())

        nsteps = tf.shape(output)[1]
        output = tf.reshape(output, [-1, 2 * self.hidden_layer])
        pred = tf.matmul(output, W) + b
        self.logits = tf.reshape(pred, [-1, nsteps, self.nlabels])
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # Optimizer.
        _lr_m = lr_method.lower()
        # global_step = tf.Variable(0)
        learning_rate = lr
        # learning_rate = tf.train.exponential_decay(
        #                 lr, global_step, 5000, 0.1, staircase=True)

        if _lr_m == 'adam':  # sgd method
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif _lr_m == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif _lr_m == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif _lr_m == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))

        # clip 1.25
        if clip < 0:
            self.opt = optimizer.minimize(self.loss)
        else:
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip)
            self.opt = optimizer.apply_gradients(zip(gradients, v))

            # Predictions.


    def train(self, train_data, tags, label, rel_loc, nepochs, batch_size, with_val_file=None):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        with tf.device("/gpu:10"):
            self._build_tf(batch_size, self.alg_optim, self.learn_rate, self.clip_val)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            # batch_size = batch_size
            # nbatches = (len(train) + batch_size - 1) // batch_size

            print('Initialized')
            mean_loss = 0
            for epchs in range(nepochs):
                batch_seq, labels = self.next_batch(batch_size, train_data, tags, rel_loc, label)
                feed_dict, _ = self.get_feed_dict(self.pad_tok, batch_seq, labels)
                _, train_loss = self.sess.run(
                    [self.opt, self.loss], feed_dict=feed_dict)
                mean_loss += train_loss
                if (epchs ) % 400 == 0 and epchs != 0:
                    print(epchs, " ", "mean train loss: ", mean_loss / (epchs + 1))
                    if with_val_file:
                        self.run_evaluate(100, with_val_file)

                if epchs % 50 == 0:
                    print(epchs, " ", "train loss: ", train_loss)

                if ( epchs ) % 400 == 0 and epchs != 0:
                    self.save_session("tmp_200_lr0.02_model_" + str(epchs))


    def retrain(self, dir_model, train_data, tags, label, rel_loc, nepochs, batch_size, with_val_file=None):
        self._build_tf(batch_size, self.alg_optim, self.learn_rate, self.clip_val)
        if self.restore == 0:
            self.restoreModel(dir_model)
        self.sess.run(tf.global_variables_initializer())
        # batch_size = batch_size
        # nbatches = (len(train) + batch_size - 1) // batch_size

        print('Initialized')
        mean_loss = 0
        for epchs in range(nepochs):
            batch_seq, labels = self.next_batch(batch_size, train_data, tags, rel_loc, label)
            feed_dict, _ = self.get_feed_dict(self.pad_tok, batch_seq, labels)
            _, train_loss = self.sess.run(
                [self.opt, self.loss], feed_dict=feed_dict)
            mean_loss += train_loss
            if epchs % 500 == 0:
                print("mean train loss: ", mean_loss / (epchs + 1))
                if with_val_file:
                    self.run_evaluate(100, with_val_file+ "_" + str(epchs))
            if epchs % 50 == 0:
                print("train loss: ", train_loss)

    # predict
    def predict(self, test, tags, rel_loc, batch_size):
        # assert self.unknown_words is not None
        with tf.device("/cpu:0"):
            self._build_tf(batch_size, self.alg_optim, self.learn_rate, self.clip_val)
            feed_dict = dict()
            batch_test = self.load_test(test, tags, rel_loc)
            nbatches = len(test) // batch_size
            viterbi_sequences, seq_lengths = self._predict(nbatches, batch_size, batch_test)
            return viterbi_sequences, seq_lengths


    def _predict_batch(self, batch_data):
        viterbi_sequences = []
        feed_dict, sequence_lengths = self.get_feed_dict(self.pad_tok, batch_data)
        logits, trans_params = self.sess.run(
            [self.logits, self.trans_params], feed_dict=feed_dict)
        # return decodeing
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths


    def _predict(self, nbatches, batch_size, data):
        viterbi_sequences = []
        seq_lengths = []
        begin = 0
        for it in range(nbatches):
            begin = it * batch_size
            batch_data = data[begin: begin + batch_size]
            vs, sl = self._predict_batch(batch_data)
            seq_lengths.extend(sl)
            viterbi_sequences.extend(vs)

        if begin + batch_size < len(data):
            batch_data = data[begin + batch_size: len(data)]
            vs, sl = self._predict_batch(batch_data)
            seq_lengths.extend(sl)
            viterbi_sequences.extend(vs)

        return viterbi_sequences, seq_lengths


    def restoreModel(self, dir_model):
        self.saver = tf.train.import_meta_graph(dir_model + ".meta")
        self.sess = tf.Session()
        with open(dir_model + ".unknown", 'rb') as f:
            self.unknown_words = pickle.load(f)
        self.saver.restore(self.sess, dir_model)
        self.restore = 1


    def save_session(self, dir_model):
        """Saves session = weights"""
        with open(dir_model + ".unknown", 'wb') as f:
            pickle.dump(self.unknown_words, f)
        self.saver.save(self.sess, dir_model)


    def close_session(self):
        """Closes the session"""
        self.sess.close()

    # run_evaluate
    def run_evaluate(self, batch_size, filename):
        assert len(self.test_data) > 0 and len(self.test_raw_data) > 0
        print("Begin Run evaluate: ")
        nbatches = len(self.test_data) // batch_size
        viterbi_sequences, seq_lengths = self._predict(nbatches, batch_size, self.test_data)
        self.evaluate("pred_200_lr0.02", filename, viterbi_sequences,
                      seq_lengths, self.test_raw_data, self.test_raw_tags)


    def evaluate(self, filename, dev_file, viterbi_seq, seq_length, val_data, val_tags, ref_seq=None):
        vit_dic = self.label_to_tag
        assert len(viterbi_seq) == len(seq_length) == len(val_data) == len(val_tags)
        with open(filename, 'w', encoding="utf-8") as f:
            for num in range(len(viterbi_seq)):
                seq = viterbi_seq[num]
                sl = seq_length[num]
                vt = val_tags[num]
                vd = val_data[num]
                if ref_seq:
                    rel = ref_seq[num]
                str = ""
                for it in range(sl - 1):
                    if ref_seq and vit_dic[seq[it]] == 'rel' :
                        if vd[it] != rel:
                            vd[it] = 'O'
                    if ref_seq and vd[it] == rel:
                        if vit_dic[seq[it]] != 'rel':
                            seq[it] = 1
                    str += "%s/%s/%s " % (vd[it], vt[it], vit_dic[seq[it]])
                str += "%s/%s/%s\n" % (vd[sl - 1], vt[sl - 1], vit_dic[seq[sl - 1]])
                f.write(str)

        # print(calc_f1(filename, dev_file)) F: 0.6530078465562337




