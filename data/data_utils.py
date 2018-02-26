import numpy as np

import gzip
import json
from util import load_dict

# Extra vocabulary symbols
_GO = '_GO'
EOS = '_EOS' # also function as PAD
UNK = '_UNK'

extra_tokens = [_GO, EOS, UNK]

pad_token = 0
start_token = 1	# start_token = 0
end_token = 2	# end_token = 1
unk_token = 3
default_token = 5

MAX_WORD_SIZE = 16
MAX_CHAR_SIZE = 96

#Luchhese
#MAX_WORD_SIZE = 20
#MAX_CHAR_SIZE = 100

with open('session_data/dict/dict_char.json', 'r') as json_file:
    dict_char = json.load(json_file)
    inv_dict_char = {}
    for words, idx in dict_char.iteritems():
        inv_dict_char[idx] = words

with open('session_data/dict/dict_word.json', 'r') as json_file:
    dict_word = json.load(json_file)
    inv_dict_word = {}
    for words, idx in dict_word.iteritems():
        inv_dict_word[idx] = words

with open('session_data/dict/dict_click.json', 'r') as json_file:
    dict_click = json.load(json_file)
    inv_dict_click = {}
    for words, idx in dict_click.iteritems():
        inv_dict_click[idx] = words

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def load_inverse_dict(dict_path):
    orig_dict = load_dict(dict_path)
    idict = {}
    for words, idx in orig_dict.iteritems():
        idict[idx] = words
    return idict


def seq2words(seq, inverse_target_dictionary):
    words = []
    for w in seq:
        if w == end_token:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append(UNK)
    return ' '.join(words)


# batch preparation of a given sequence
def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)
    
    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * end_token
    
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths


# batch preparation of a given sequence pair for training
def prepare_train_batch(x, y, maxlen=None):
    lengths_x = [len(s) for s in x]
    lengths_y = [len(s) for s in y]

    return x, lengths_x, y, lengths_y


def num2word(num_list):
    word_list = []
    for i in num_list:
        try:
            word_list.append(inv_dict_word[i])
        except:
            continue
    return " ".join(word_list)

def word_seq_list(seq):
    res = []
    for i in seq:
        res.append(num2word(i))
    return res

def print_idx(c):
    print "----"
    n = 1
    for i in c:
        print str(n) + "\t" + str(i)
        n = n + 1

def print_group_idx(cs):
    print "----"
    for i in range(len(cs[0])):
        res = str(i+1)
        for j in cs:
            res += "\t" + str(j[i])
        print res
