
#!/usr/bin/env python
# coding: utf-8

import os
import math
import time
import json
import random

from collections import OrderedDict

import numpy as np
import tensorflow as tf

from data.data_iterator import *

import data.util as util
import data.data_utils as data_utils
from data.data_utils import prepare_batch
from data.data_utils import prepare_train_batch
from data.data_utils import word_seq_list
from data.data_utils import print_idx
from data.data_utils import print_group_idx

from seq2seq_model import Seq2SeqModel

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 0, 'Beam width used in beamsearch')
tf.app.flags.DEFINE_integer('decode_batch_size', 20, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 2, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', None, 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('decode_input', 'session_data/webis-smc-12.data', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', 'data/newstest2012.bpe.de.trans', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

def evaluate(predict, groud_truth):
    TP = TN = FP = FN = 0
    for i, _ in enumerate(predict):
        p = predict[i]
        g = groud_truth[i]

        if p == g:
            if g == 1:
                TP += 1
            if g == 0:
                TN += 1
        else:
            if g == 1:
                FP += 1
            if g == 0:
                FN += 1
    print "==== EVALUATE ===="
    print "TP, TN, FP, FN"
    print TP, TN, FP, FN

    P1 = float(TP)/(TP+FP)
    R1 = float(TP)/(TP+FN)
    F1 = 2*P1*R1/(P1+R1)

    P2 = float(TN)/(TN+FN)
    R2 = float(TN)/(TN+FP)
    F2 = 2*P2*R2/(P2+R2)


    print "True Task:"
    print "P:%.3f    R:%.3f    F:%.3f" % (P1, R1, F1)
    print "False Task:"
    print "P:%.3f    R:%.3f    F:%.3f" % (P2, R2, F2)
    print "==== END ===="
    return 0


def load_config(FLAGS):
    
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % FLAGS.model_path, 'rb')))
    for key, value in FLAGS.__flags.items():
        config[key] = value

    return config


def load_model(session, config):
    
    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print 'Reloading model parameters..'
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():
    # Load model config
    _data_word = None
    _alignment = None
    _predict = np.array([])
    _ground_truth = np.array([])
    config = load_config(FLAGS)

    # Load source data to decode
    #test_set = TextIterator(source=config['decode_input'],
                            #batch_size=config['decode_batch_size'],
                            #source_dict=config['source_vocabulary'],
                            #maxlen=None,
                            #n_words_source=config['num_encoder_symbols'])

    test_set = SegTextIterator(source=config['decode_input'],
                               time_step=config['time_step'],
                               batch_size=config['batch_size'], set_type=1, test_set_partition=config["test_set_partition"], balance=config["data_balance"], noise=config["noise"])

    # Load inverse dictionary used in decoding
    #target_inverse_dict = data_utils.load_inverse_dict(config['target_vocabulary'])
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print 'Decoding {}..'.format(FLAGS.decode_input)
            #if FLAGS.write_n_best:
            #    fout = [data_utils.fopen(("%s_%d" % (FLAGS.decode_output, k)), 'w') \
            #            for k in range(FLAGS.beam_width)]
            #else:
            #    fout = [data_utils.fopen(FLAGS.decode_output, 'w')]
            step = 0
            set_length = len(test_set)
            for source_data, source_word, source_char, source_word_mask, source_char_mask, target_seq, true_batch_size in test_set:
                source_data, source_len, target, target_len = prepare_train_batch(source_data, target_seq)
                source = source_data, source_word, source_word_mask, source_char, source_char_mask

                #source, source_len = prepare_batch(source_seq)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids, alignment, _ = model.predict(sess, encoder_inputs=source, 
                                              encoder_inputs_length=source_len)
                a = _[10:1010, :]
                for i in a:
                    t = []
                    for j in i:
                        t.append(str(float(j)))
                    print '\t'.join(t)
                exit()
                if config['use_attention']:
                    try:
                        _data_word = np.concatenate((_data_word, source_word))
                    except:
                        _data_word = source_word
                    try:
                        _alignment = np.concatenate((_alignment, alignment))
                    except:
                        _alignment = alignment
                _predict = np.concatenate((_predict, predicted_ids[:, 0, 0]))
                _ground_truth = np.concatenate((_ground_truth, target[:, 0]))

                progress = 100 * float(step) / set_length
                print "progress %.1f%%" % progress
                step += 1

                # Write decoding results
                #for k, f in reversed(list(enumerate(fout))):
                #    for seq in predicted_ids:
                #        f.write(str(data_utils.seq2words(seq[:,k], target_inverse_dict)) + '\n')
                #    if not FLAGS.write_n_best:
                #        break
                #print '  {}th line decoded'.format(idx * FLAGS.decode_batch_size)



            if config['use_attention']:
                idxs = [j for j in range(len(_alignment))]
                for idx in idxs:
                    print ">>>>"
                    print_group_idx([_alignment[idx], word_seq_list(_data_word[idx])])
                    #print_idx(word_seq_list(_data_word[idx]))
                    #print_idx(_alignment[idx])
                    print _predict[idx], _ground_truth[idx]

            evaluate(_predict, _ground_truth)
            print 'Decoding terminated'
        except IOError:
            pass


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()

