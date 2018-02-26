#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

import data.data_utils as data_utils

class Seq2SeqModel(object):

    def __init__(self, config, mode):

        assert mode.lower() in ['train', 'decode']

        self.config = config
        self.mode = mode.lower()

        self.cell_type = config['cell_type']
        self.hidden_units = config['hidden_units']
        self.decoder_hidden_units = config['hidden_units']
        #self.decoder_hidden_units = int(config['hidden_units']) * 2
        #self.hidden_units = config['embedding_size'] + config['feature_size']
        
        self.depth = config['depth']
        self.attention_type = config['attention_type']
        self.embedding_size = config['embedding_size']
        self.decoder_embedding_size = config['decoder_embedding_size']
        #self.bidirectional = config.bidirectional

        self.batch_size = config['batch_size']
        self.time_step = config['time_step']
        self.feature_size = config['feature_size']
        self.feature_seq_size = config['feature_seq_size']
        self.feature_seq_c_size = config['feature_seq_c_size']
       
        self.num_word_symbols = config['num_word_symbols']
        self.num_char_symbols = config['num_char_symbols']
        self.num_decoder_symbols = config['num_decoder_symbols']

        self.use_attention = config['use_attention']
        self.use_rnn_embedding = config['use_rnn_embedding']
        self.use_residual = config['use_residual']
        self.attn_input_feeding = config['attn_input_feeding']
        self.use_dropout = config['use_dropout']
        self.keep_prob = 1.0 - config['dropout_rate']

        self.optimizer = config['optimizer']
        self.learning_rate = config['learning_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
	    tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')

        self.use_beamsearch_decode=False 
        if self.mode == 'decode':
            self.beam_width = config['beam_width']
            self.use_beamsearch_decode = True if self.beam_width > 1 else False
            self.max_decode_step = config['max_decode_step']

        self.load_embedding()
        self.build_model()

       
    def build_model(self):
        print("building model..")

        # Building encoder and decoder networks
        self.init_placeholders()
        self.build_encoder()
        #self.build_decoder()
        self.build_upper()

        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()

        #self.debug_output = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[6]
        #self.debug_output = tf.Print(self.debug_output, [self.debug_output], summarize=1000)
        #print tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        #self.debug_output = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)[0]


    def init_placeholders(self):       
        # encoder_inputs: [batch_size, max_time_steps, all_feature_size]
        #self.encoder_inputs = tf.placeholder(dtype=tf.int32,
            #shape=(None, None), name='encoder_inputs')

        self.encoder_data = tf.placeholder(dtype=self.dtype,
            shape=(self.batch_size, self.time_step, self.feature_size), name='encoder_data')

        ### WORD seq
        self.encoder_seq = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.time_step, self.feature_seq_size), name='encoder_seq')

        self.encoder_seq_mask = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.time_step), name='encoder_seq_mask')

        ### CHAR seq
        self.encoder_seq_c = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.time_step, self.feature_seq_c_size), name='encoder_seq_c')

        self.encoder_seq_mask_c = tf.placeholder(dtype=tf.int32,
            shape=(self.batch_size, self.time_step), name='encoder_seq_mask_c')

        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=(None,), name='encoder_inputs_length')

        # get dynamic batch_size
        #self.batch_size = self.encoder_data.get_shape().as_list()[0]
        if self.mode == 'train':

            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=(self.batch_size, None), name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.start_token
            decoder_end_token = tf.ones(
                shape=[self.batch_size, 1], dtype=tf.int32) * data_utils.end_token            

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                  self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                   decoder_end_token], axis=1)

    def load_embedding(self):
        self.init_embedding = []
        with open("session_data/init_embedding") as f:
            for line in f.readlines():
                line = [float(j) for j in line.strip().split(' ')]
                self.init_embedding.append(line)

    def attention(self, inputs, attention_size, return_alphas=False):
        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
        alphas = tf.nn.softmax(vu)              # (B,T) shape also

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def build_encoder(self):
        print("building encoder..")
        with tf.variable_scope('encoder'):
            random_initializer = tf.contrib.layers.xavier_initializer(seed=0, dtype=self.dtype)
            constant_initializer = tf.constant_initializer(self.init_embedding)
            
            ### word embedding ###
            self.word_embedding_cell = self.build_one_cell(self.embedding_size)
            self.word_embeddings = tf.get_variable(name='word_embedding',
                shape=[self.num_word_symbols, self.embedding_size],
                initializer=constant_initializer, dtype=self.dtype)

            if not self.use_rnn_embedding:
                sequence = []
                sequence_lengths = []
                for i in range(self.batch_size):
                    for j in range(self.time_step):
                        c = self.encoder_seq[i, j, :]
                        mask_indicator = self.encoder_seq_mask[i, j]
                        mask = tf.sequence_mask([mask_indicator], self.feature_seq_size)[0]
                        c = tf.boolean_mask(c, mask)
                        sequence.append(c)
                        sequence_lengths.append(mask_indicator)

                embedding_sequence = []
                for i in range(len(sequence)):
                    embed = tf.nn.embedding_lookup(params=self.word_embeddings, ids=sequence[i])
                    embed = tf.reduce_mean(embed, axis=0)
                    #embed = tf.reduce_max(embed, axis=0)

                    embedding_sequence.append(embed)

                embedding_sequence_tensor = tf.convert_to_tensor(embedding_sequence)
                batch_words_embeddings = tf.reshape(embedding_sequence_tensor, [self.batch_size, self.time_step, self.embedding_size])

            elif self.use_rnn_embedding:
                words_sequence = tf.reshape(self.encoder_seq, [self.batch_size * self.time_step, self.feature_seq_size])
                words_sequence_mask = tf.reshape(self.encoder_seq_mask, [self.batch_size * self.time_step])

                words_sequence_embeddings = tf.nn.embedding_lookup(params=self.word_embeddings, ids=words_sequence)

                self.word_outputs, self.word_last_states = tf.nn.dynamic_rnn(
                    cell=self.word_embedding_cell, 
                    inputs=words_sequence_embeddings,
                    sequence_length=words_sequence_mask, 
                    dtype=self.dtype,
                    time_major=False,
                    scope="word_embedding")
                batch_words_embeddings = tf.reshape(self.word_last_states[0].h, [self.batch_size, self.time_step, self.embedding_size])



            ### char embedding ###
            self.char_embedding_cell = self.build_one_cell(self.embedding_size)
            self.char_embeddings = tf.get_variable(name='char_embedding',
                shape=[self.num_char_symbols, self.embedding_size],
                initializer=random_initializer, dtype=self.dtype)

            chars_sequence = tf.reshape(self.encoder_seq_c, [self.batch_size * self.time_step, self.feature_seq_c_size])
            chars_sequence_mask = tf.reshape(self.encoder_seq_mask_c, [self.batch_size * self.time_step])

            chars_sequence_embeddings = tf.nn.embedding_lookup(params=self.char_embeddings, ids=chars_sequence)

            self.char_outputs, self.char_last_states = tf.nn.dynamic_rnn(
                cell=self.char_embedding_cell, 
                inputs=chars_sequence_embeddings,
                sequence_length=chars_sequence_mask, 
                dtype=self.dtype,
                time_major=False,
                scope="char_embedding")
            batch_chars_embeddings = tf.reshape(self.char_last_states[0].h, [self.batch_size, self.time_step, self.embedding_size])



            ####################
            ### feed to lstm ###
            input_layer = Dense(self.hidden_units, dtype=self.dtype, name='input_projection')
            input_layer_hidden_size = Dense(self.hidden_units, dtype=self.dtype, name='input_projection_hidden_size')
            self.encoder_inputs_embedded = input_layer(self.encoder_data)
            self.encoder_cell = self.build_encoder_cell()

            self.encoder_outputs, _ = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, 
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, 
                dtype=self.dtype,
                time_major=False,
                scope="main_dynamic_rnn")


            w_input_layer = Dense(self.hidden_units, dtype=self.dtype, name='w_input_projection')
            w_input_layer_hidden_size = Dense(self.hidden_units, dtype=self.dtype, name='w_input_projection_hidden_size')
            self.w_encoder_inputs_embedded = w_input_layer(batch_words_embeddings)
            self.w_encoder_cell = self.build_encoder_cell()

            self.w_encoder_outputs, _ = tf.nn.dynamic_rnn(
                cell=self.w_encoder_cell, 
                inputs=self.w_encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=self.dtype,
                time_major=False,
                scope="w_main_dynamic_rnn")
            
            c_input_layer = Dense(self.hidden_units, dtype=self.dtype, name='c_input_projection')
            c_input_layer_hidden_size = Dense(self.hidden_units, dtype=self.dtype, name='c_input_projection_hidden_size')
            self.c_encoder_inputs_embedded = c_input_layer(batch_chars_embeddings)
            self.c_encoder_cell = self.build_encoder_cell()

            self.c_encoder_outputs, _ = tf.nn.dynamic_rnn(
                cell=self.c_encoder_cell, 
                inputs=self.c_encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=self.dtype,
                time_major=False,
                scope="c_main_dynamic_rnn")


    def build_upper(self):
        #this_output = tf.reduce_mean(self.encoder_outputs, axis=1)
        #this_output = self.encoder_outputs[:, -1, :]
        this_output, self.alignment = self.attention(tf.concat([self.encoder_outputs, self.w_encoder_outputs, self.c_encoder_outputs], axis=2), self.hidden_units, True)
        #this_output, self.alignment = self.attention(self.encoder_outputs, self.hidden_units, True)

        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_units*3, 2], dtype=self.dtype)
        softmax_b = tf.get_variable("softmax_b", [self.batch_size ,2], dtype=self.dtype)

        logits = tf.matmul(this_output, softmax_w) + softmax_b
        
        if self.mode == 'train':
            labels = tf.reshape(self.decoder_inputs, [self.batch_size])

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+1e-10, labels=labels)
            self.loss = tf.reduce_mean(self.loss)
            
            tf.summary.scalar('loss', self.loss)
            self.init_optimizer()

        elif self.mode == 'decode':
            self.decoder_pred_decode = tf.argmax(logits, axis=1)
            self.decoder_pred_decode = tf.reshape(self.decoder_pred_decode, [self.batch_size, 1, 1])

        

    def build_decoder(self):
        print("building decoder and attention..")
        with tf.variable_scope('decoder'):
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()

            initializer = tf.contrib.layers.xavier_initializer(seed=0, dtype=self.dtype)
            
            self.decoder_embeddings = tf.get_variable(name='embedding',
                shape=[self.num_decoder_symbols, self.decoder_embedding_size],
                initializer=initializer, dtype=self.dtype)

            input_layer = Dense(self.decoder_hidden_units, dtype=self.dtype, name='input_projection')
            output_layer = Dense(self.num_decoder_symbols, name='output_projection')

            if self.mode == 'train':
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.decoder_embeddings, ids=self.decoder_inputs_train)
               
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                   sequence_length=self.decoder_inputs_length_train,
                                                   time_major=False,
                                                   name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                   helper=training_helper,
                                                   initial_state=self.decoder_initial_state,
                                                   output_layer=output_layer)
                                                   #output_layer=None)
                    
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                (self.decoder_outputs_train, self.decoder_last_state_train, 
                 self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(
                    decoder=training_decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=max_decoder_length))
                 
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output) 
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train, axis=-1,
                                                    name='decoder_pred_train')

                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train, 
                                         maxlen=max_decoder_length, dtype=self.dtype, name='masks')

                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train, 
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True,)

                tf.summary.scalar('loss', self.loss)

                # Contruct graphs for minimizing loss
                self.init_optimizer()

            elif self.mode == 'decode':
        
                # Start_tokens: [batch_size,] `int32` vector
                start_tokens = tf.ones([self.batch_size,], tf.int32) * data_utils.start_token
                end_token = data_utils.end_token

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.decoder_embeddings, inputs))
                    
                if not self.use_beamsearch_decode:
                    # Helper to feed inputs for greedy decoding: uses the argmax of the output
                    decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                                    end_token=end_token,
                                                                    embedding=embed_and_input_proj)
                    # Basic decoder performs greedy decoding at each time step
                    print("building greedy decoder..")
                    inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                             helper=decoding_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=output_layer)
                else:
                    # Beamsearch is used to approximately find the most likely translation
                    print("building beamsearch decoder..")
                    inference_decoder = beam_search_decoder.BeamSearchDecoder(cell=self.decoder_cell,
                                                               embedding=embed_and_input_proj,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=self.decoder_initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=output_layer,)
                # For GreedyDecoder, return
                # decoder_outputs_decode: BasicDecoderOutput instance
                #                         namedtuple(rnn_outputs, sample_id)
                # decoder_outputs_decode.rnn_output: [batch_size, max_time_step, num_decoder_symbols] 	if output_time_major=False
                #                                    [max_time_step, batch_size, num_decoder_symbols] 	if output_time_major=True
                # decoder_outputs_decode.sample_id: [batch_size, max_time_step], tf.int32		if output_time_major=False
                #                                   [max_time_step, batch_size], tf.int32               if output_time_major=True 
                
                # For BeamSearchDecoder, return
                # decoder_outputs_decode: FinalBeamSearchDecoderOutput instance
                #                         namedtuple(predicted_ids, beam_search_decoder_output)
                # decoder_outputs_decode.predicted_ids: [batch_size, max_time_step, beam_width] if output_time_major=False
                #                                       [max_time_step, batch_size, beam_width] if output_time_major=True
                # decoder_outputs_decode.beam_search_decoder_output: BeamSearchDecoderOutput instance
                #                                                    namedtuple(scores, predicted_ids, parent_ids)

                (self.decoder_outputs_decode, self.decoder_last_state_decode,
                 self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=False,
                    #impute_finished=True,	# error occurs
                    maximum_iterations=self.max_decode_step))

                ### get alignment from decoder_last_state
                if self.use_attention:
                    self.alignment = self.decoder_last_state_decode[0].alignment_history.stack()
                else:
                    self.alignment = []

                if not self.use_beamsearch_decode:
                    # decoder_outputs_decode.sample_id: [batch_size, max_time_step]
                    # Or use argmax to find decoder symbols to emit:
                    # self.decoder_pred_decode = tf.argmax(self.decoder_outputs_decode.rnn_output,
                    #                                      axis=-1, name='decoder_pred_decode')

                    # Here, we use expand_dims to be compatible with the result of the beamsearch decoder
                    # decoder_pred_decode: [batch_size, max_time_step, 1] (output_major=False)
                    self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

                else:
                    # Use beam search to approximately find the most likely translation
                    # decoder_pred_decode: [batch_size, max_time_step, beam_width] (output_major=False)
                    self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids


    def build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)
            
        return cell

    def _build_single_cell(self):
        cell_type = LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = GRUCell
        cell = cell_type(self.decoder_hidden_units)

        if self.use_dropout:
            cell = DropoutWrapper(cell, dtype=self.dtype,
                                  output_keep_prob=self.keep_prob_placeholder,)
        if self.use_residual:
            cell = ResidualWrapper(cell)
            
        return cell

    def build_cell(self, hidden_size):
        cell_type = LSTMCell
        cell = cell_type(hidden_size)
        return cell


    # Building encoder cell
    def build_encoder_cell (self):
        return MultiRNNCell([self.build_single_cell() for i in range(self.depth)])

    def build_one_cell(self, hidden_size):
        return MultiRNNCell([self.build_cell(hidden_size)])

    # Building decoder cell and attention. Also returns decoder_initial_state
    def build_decoder_cell(self):

        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length

        # To use BeamSearchDecoder, encoder_outputs, encoder_last_state, encoder_inputs_length 
        # needs to be tiled so that: [batch_size, .., ..] -> [batch_size x beam_width, .., ..]
        if self.use_beamsearch_decode:
            print ("use beamsearch decoding..")
            encoder_outputs = seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_last_state = nest.map_structure(
                lambda s: seq2seq.tile_batch(s, self.beam_width), self.encoder_last_state)
            encoder_inputs_length = seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)

        with tf.variable_scope('attention_mechanism_Bahdanau'):
            # Building attention mechanism: Default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            self.attention_mechanism = attention_wrapper.BahdanauAttention(
                num_units=self.decoder_hidden_units, memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length, name="Bahdanau") 

        with tf.variable_scope('attention_mechanism_Luong'):
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            if self.attention_type.lower() == 'luong':
                self.attention_mechanism = attention_wrapper.LuongAttention(
                    num_units=self.decoder_hidden_units, memory=encoder_outputs, 
                    memory_sequence_length=encoder_inputs_length, name="Luong")
 
        # Building decoder_cell
        self.decoder_cell_list = [self._build_single_cell() for i in range(self.depth)]
        decoder_initial_state = encoder_last_state

        if not self.use_attention:
            return MultiRNNCell(self.decoder_cell_list), decoder_initial_state

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs

            # Essential when use_residual=True
            _input_layer = Dense(self.decoder_hidden_units, dtype=self.dtype,
                                 name='attn_input_feeding')
            return _input_layer(array_ops.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = attention_wrapper.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.decoder_hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=True,
            name='Attention_Wrapper')

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        # Also if beamsearch decoding is used, the batch_size argument in .zero_state
        # should be ${decoder_beam_width} times to the origianl batch_size
        batch_size = self.batch_size if not self.use_beamsearch_decode \
                     else self.batch_size * self.beam_width
        initial_state = [state for state in encoder_last_state]

        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
          batch_size=batch_size, dtype=self.dtype)
        decoder_initial_state = tuple(initial_state)

        return MultiRNNCell(self.decoder_cell_list), decoder_initial_state


    def init_optimizer(self):
        print("setting optimizer..")
        # Gradients and SGD update operation for training the model
        trainable_params = tf.trainable_variables()
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        # Update the model
        self.updates = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def save(self, sess, path, var_list=None, global_step=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)

        # temporary code
        #del tf.get_collection_ref('LAYER_NAME_UIDS')[0]
        save_path = saver.save(sess, save_path=path, global_step=global_step)
        print('model saved at %s' % save_path)
        

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


    def train(self, sess, encoder_inputs, encoder_inputs_length, 
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob
 
        output_feed = [self.updates,	# Update Op that does optimization
                       self.loss,	# Loss for current batch
                       self.summary_op] # Training summary
                       #self.debug_output]
        
        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]	# loss, summary


    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length,
                                      decoder_inputs, decoder_inputs_length, False)
        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0

        output_feed = [self.loss,	# Loss for current batch
                       self.summary_op]	# Training summary
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1]	# loss


    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        
        input_feed = self.check_feeds(encoder_inputs, encoder_inputs_length, 
                                      decoder_inputs=None, decoder_inputs_length=None, 
                                      decode=True)

        # Input feeds for dropout
        input_feed[self.keep_prob_placeholder.name] = 1.0
 
        output_feed = [self.decoder_pred_decode, self.alignment, self.word_embeddings]
        outputs = sess.run(output_feed, input_feed)

				# GreedyDecoder: [batch_size, max_time_step]
        return outputs[0], outputs[1], outputs[2] # BeamSearchDecoder: [batch_size, max_time_step, beam_width]


    def check_feeds(self, encoder_inputs, encoder_inputs_length, 
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size] 
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """ 

        input_feed = {}

        encoder_data, encoder_seq, encoder_seq_mask, encoder_seq_c, encoder_seq_mask_c = encoder_inputs
    
        input_feed[self.encoder_data.name] = encoder_data
        input_feed[self.encoder_seq.name] = encoder_seq
        input_feed[self.encoder_seq_mask.name] = encoder_seq_mask
        input_feed[self.encoder_seq_c.name] = encoder_seq_c
        input_feed[self.encoder_seq_mask_c.name] = encoder_seq_mask_c
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed 
