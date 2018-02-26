
import numpy as np
import random
import shuffle
from util import load_dict

import data_utils

'''
Much of this code is based on the data_iterator.py of
nematus project (https://github.com/rsennrich/nematus)
'''

class TextIterator:
    """Simple Text iterator."""
    def __init__(self, source, source_dict,
                 batch_size=128, maxlen=None,
                 n_words_source=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=20,
                 ):

        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')

        self.source_dict = load_dict(source_dict)
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source = shuffle.main([self.source_orig], temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []

        # fill buffer, if it's empty
        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip().split())
    
            # sort by buffer
            if self.sort_by_length:
                slen = np.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
    
                _sbuf = [self.source_buffer[i] for i in sidx]
    
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()
    
        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
    
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]
    
                if self.maxlen and len(ss) > self.maxlen:
                    continue
                if self.skip_empty and (not ss):
                    continue
                source.append(ss)
    
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
    
        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0:
            source = self.next()
    
        return source

class BiTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 maxibatch_size=20):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = data_utils.fopen(source, 'r')
            self.target = data_utils.fopen(target, 'r')

        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else data_utils.unk_token for w in ss]

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict 
                      else data_utils.unk_token for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target 
                          else data_utils.unk_token for w in tt]

                if self.maxlen:
                    if len(ss) > self.maxlen and len(tt) > self.maxlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target

class SegTextIterator:
    """Simple Text iterator."""
    def __init__(self, source, time_step=10, batch_size=20, set_type=0, seed=0, test_set_partition=0.05, balance=False, noise=0):
        np.set_printoptions(threshold='nan')
        self.source = data_utils.fopen(source, 'r')
        self.time_step = time_step
        self.batch_size = batch_size
        self.iterator = 0
        self.end_of_data = False

        random.seed(seed)
        self.load(source, time_step, set_type, test_set_partition, balance, noise)

    def load(self, source, seq_length, set_type, test_set_partition, balance, noise):
        self.data_buffer = []
        self.word_buffer = []
        self.char_buffer = []
        self.word_mask_buffer = []
        self.char_mask_buffer = []
        self.target_buffer = []

        d = {}
        SEQ_SIZE = seq_length
        with open(source) as ff:
            for line_number, line in enumerate(ff):
                line = line.strip().split("\t")


                part1 = part2 = part3 = []
                for i in range(5):
                    part1.append(int(line[i]))
                part2 = [ int(i) for i in line[5].split(",")]
                part3 = [ int(i) for i in line[6].split(",")]

                user_id = line[0]
                if user_id not in d:
                    d[user_id] = []
                d[user_id].append([part1, part2, part3])

        buff = []
        for user_id, content in d.items():
            for i in range(len(content) - 1):
                data = []
                word = []
                char = []
                mask_word = []
                mask_char = []
                for j in range(SEQ_SIZE):
                    a, b, c, mb, mc = self.get_element(content, i - (SEQ_SIZE/2 - 1) + j, i, noise)
                    #a, b, c, mb, mc = self.get_element(content, i - (SEQ_SIZE/2 - 1) + j)
                    #a, b, c, mb, mc = self.get_element(content, i - (SEQ_SIZE - 2) + j)
                    data.append(a)
                    word.append(b)
                    char.append(c)
                    mask_word.append(mb)
                    mask_char.append(mc)
                tar = np.array([0])
                if content[i][0][1] == content[i+1][0][1]:
                    tar = np.array([1])
                
                buff.append((data, word, char, mask_word, mask_char, tar))

        if balance:
            p = 0
            for i in buff:
                a,b,c,d,e,tar = i
                if tar[0] == 0:
                    p += 1
            new_buff = []
            for i in buff:
                a,b,c,d,e,tar = i
                if tar[0] == 0:
                    new_buff.append(i)
                if tar[0] == 1 and p > 0:
                    new_buff.append(i)
                    p -= 1
            buff = new_buff

        random.shuffle(buff)
        pos = int(len(buff) * (1 - test_set_partition))
        if set_type == 0:
            buff = buff[:pos]
        elif set_type == 1:
            buff = buff[pos:]

        for data, word, char, mask_word, mask_char, tar in buff:
            self.data_buffer.append(data)
            self.word_buffer.append(word)
            self.char_buffer.append(char)
            self.word_mask_buffer.append(mask_word)
            self.char_mask_buffer.append(mask_char)
            self.target_buffer.append(tar)
        
    
    def get_element(self, pool, i, anchor=0, noise=0):
        #max word size 16, max char size 96
        #padding = np.array([-1, -1, -1] + [data_utils.default_token for m in range(data_utils.MAX_WORD_SIZE)] + [data_utils.default_token for m in range(data_utils.MAX_CHAR_SIZE)])
        padding = [[-1, -1, -1], 
                [data_utils.default_token for m in range(data_utils.MAX_WORD_SIZE)], 
                [data_utils.default_token for m in range(data_utils.MAX_CHAR_SIZE)]]

        if i >= 0 and i < len(pool):
            word = [data_utils.pad_token for m in range(data_utils.MAX_WORD_SIZE)]
            char = [data_utils.pad_token for m in range(data_utils.MAX_CHAR_SIZE)]
            for n, j1 in enumerate(pool[i][1]):
                word[n] = j1
            for n, j2 in enumerate(pool[i][2]):
                char[n] = j2
            if noise == 0:
                return pool[i][0][2:5], word, char, len(pool[i][1]), len(pool[i][2])
            else:
                mask = [anchor-noise/2 + 1 + j for j in range(noise)]
                if i in mask:
                    return pool[i][0][2:5], word, char, len(pool[i][1]), len(pool[i][2])
                else:
                    return padding[0], padding[1], padding[2], 1 , 1
        else:
            return padding[0], padding[1], padding[2], 1 , 1

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        self.iterator = 0

    def next(self):
        if self.iterator + self.batch_size - 1 < len(self.data_buffer):
            data = self.data_buffer[self.iterator:self.iterator+self.batch_size]
            word = self.word_buffer[self.iterator:self.iterator+self.batch_size]
            char = self.char_buffer[self.iterator:self.iterator+self.batch_size]
            word_mask = self.word_mask_buffer[self.iterator:self.iterator+self.batch_size]
            char_mask = self.char_mask_buffer[self.iterator:self.iterator+self.batch_size]
            target = self.target_buffer[self.iterator:self.iterator+self.batch_size]
        else:
            self.iterator = 0
            self.end_of_data = True
            raise StopIteration
            #if self.iterator < len(self.data_buffer):
            #    data = self.data_buffer[self.iterator:]
            #    word = self.word_buffer[self.iterator:]
            #    char = self.char_buffer[self.iterator:]
            #    word_mask = self.word_mask_buffer[self.iterator:]
            #    char_mask = self.char_mask_buffer[self.iterator:]
            #    target = self.target_buffer[self.iterator:]

        self.iterator += self.batch_size
        return np.array(data), np.array(word), np.array(char), np.array(word_mask), np.array(char_mask), np.array(target), len(data)
