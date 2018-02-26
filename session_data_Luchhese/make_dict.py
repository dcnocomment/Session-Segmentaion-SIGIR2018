#!/usr/bin/python
#-*- coding: gb2312 -*-
import math
import sys 
import os
import time
import numpy as np
import json

dict_char = {}
dict_word = {}
dict_click = {}
char_idx = 10
word_idx = 10
click_idx = 1
with open("webis-smc-12.txt") as ff:
    for line_number, line in enumerate(ff):
        if line_number == 0:
            continue
        line = line.strip().split("\t")
        if len(line) < 6:
            continue
        query = line[1]
        click = line[4]

        for i in query:
            if i not in dict_char:
                dict_char[i] = char_idx
                char_idx += 1
        for i in query.split(" "):
            if i not in dict_word:
                dict_word[i] = word_idx
                word_idx += 1
        if click not in dict_click:
            dict_click[click] = click_idx
            click_idx += 1

with open('dict/dict_char.json', 'w') as json_file:
    json_file.write(json.dumps(dict_char))
with open('dict/dict_word.json', 'w') as json_file:
    json_file.write(json.dumps(dict_word))
dict_click[""] = -1
with open('dict/dict_click.json', 'w') as json_file:
    json_file.write(json.dumps(dict_click))
