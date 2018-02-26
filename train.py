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

import data.data_utils as data_utils
from data.data_utils import prepare_batch
from data.data_utils import prepare_train_batch

from seq2seq_model import Seq2SeqModel


# Data loading parameters
tf.app.flags.DEFINE_string('source_train_data', 'session_data/webis-smc-12.data', 'Path to source training data')
tf.app.flags.DEFINE_string('source_valid_data', 'session_data/webis-smc-12.data', 'Path to source validation data')

# Network parameters
tf.app.flags.DEFINE_string('cell_type', 'lstm', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('depth', 1, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Embedding dimensions of encoder inputs')
tf.app.flags.DEFINE_integer('decoder_embedding_size', 10, 'Embedding dimensions of decoder inputs')
tf.app.flags.DEFINE_integer('num_word_symbols', 5000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_char_symbols', 200, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('num_decoder_symbols', 3, 'Target vocabulary size')

tf.app.flags.DEFINE_boolean('use_attention', True, 'Use_attention')
tf.app.flags.DEFINE_boolean('use_rnn_embedding', True, 'Use_attention')
tf.app.flags.DEFINE_boolean('use_residual', False, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('attn_input_feeding', True, 'Use input feeding method in attentional decoder')
tf.app.flags.DEFINE_boolean('use_dropout', False, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.2, 'Dropout probability for input/output/state units (0.0: no dropout)')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('feature_size', 3, 'feature_size')
tf.app.flags.DEFINE_integer('feature_seq_size', data_utils.MAX_WORD_SIZE, 'feature_seq_size')
tf.app.flags.DEFINE_integer('feature_seq_c_size', data_utils.MAX_CHAR_SIZE, 'feature_seq_c_size')
tf.app.flags.DEFINE_integer('batch_size', 40, 'Batch size')
tf.app.flags.DEFINE_integer('time_step', 10, 'Time step')
tf.app.flags.DEFINE_integer('max_epochs', 50, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 50000, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1000000, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'translate.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', True, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', True, 'Sort pre-fetched minibatches by their target sequence lengths')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')

# Data parameters
tf.app.flags.DEFINE_float('test_set_partition', 0.05, 'parti')
tf.app.flags.DEFINE_boolean('data_balance', True, 'Data_balance')
tf.app.flags.DEFINE_integer('noise', 0, 'Data_noise')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS

def create_model(session, FLAGS):

    config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2SeqModel(config, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print 'Reloading model parameters..'
        model.restore(session, ckpt.model_checkpoint_path)
        
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print 'Created new model parameters..'
        session.run(tf.global_variables_initializer())
   
    return model

def train():
    # Load parallel data to train
    print 'Loading training data..'
    train_set = SegTextIterator(source=FLAGS.source_train_data,
                               time_step=FLAGS.time_step,
                               batch_size=FLAGS.batch_size, set_type=0, test_set_partition=FLAGS.test_set_partition, balance=FLAGS.data_balance, noise=FLAGS.noise)

    if FLAGS.source_valid_data:
        print 'Loading validation data..'
        valid_set = SegTextIterator(source=FLAGS.source_valid_data,
                                   time_step=FLAGS.time_step,
                                   batch_size=FLAGS.batch_size, set_type=1, test_set_partition=FLAGS.test_set_partition, balance=FLAGS.data_balance, noise=FLAGS.noise)
    else:
        valid_set = None

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print 'Training..'
        for epoch_idx in xrange(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print 'Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs)
                break

            for source_data, source_word, source_char, source_word_mask, source_char_mask, target_seq, true_batch_size in train_set:
                source_data, source_len, target, target_len = prepare_train_batch(source_data, target_seq)
                source = source_data, source_word, source_word_mask, source_char, source_char_mask
                
                #print source_char
                #print "===="
                #print source_char_mask
                #print source_len
                #print "===="
                #print target
                #print target_len
                #print type(source_char), type(source_char_mask), type(source_len), type(target_len)
                #exit()
                if source is None or target is None:
                    print 'No samples under max_seq_length ', FLAGS.max_seq_length
                    continue

                # Execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len, 
                                                 decoder_inputs=target, decoder_inputs_length=target_len)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(source_len+target_len))
                sents_seen += float(FLAGS.batch_size) # batch_size

                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print 'Epoch ', model.global_epoch_step.eval(), 'Step ', model.global_step.eval(), \
                          'Perplexity {0:.5f}'.format(avg_perplexity), '    Step-time ', step_time, \
                          '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec)

                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if valid_set and model.global_step.eval() % FLAGS.valid_freq == 0:
                    print 'Validation step'
                    valid_loss = 0.0
                    valid_sents_seen = 0
                    for source_seq, target_seq in valid_set:
                        # Get a batch from validation parallel data
                        source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq)

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                        decoder_inputs=target, decoder_inputs_length=target_len)
                        batch_size = source.shape[0]

                        valid_loss += step_loss * batch_size
                        valid_sents_seen += batch_size
                        print '  {} samples seen'.format(valid_sents_seen)

                    valid_loss = valid_loss / valid_sents_seen
                    print 'Valid perplexity: {0:.2f}'.format(math.exp(valid_loss))

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print 'Saving the model..'
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(model.config,
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                              indent=2)

            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print 'Epoch {0:} DONE'.format(model.global_epoch_step.eval())
        
        print 'Saving the last model..'
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'wb'),
                  indent=2)
        
    print 'Training Terminated'



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
