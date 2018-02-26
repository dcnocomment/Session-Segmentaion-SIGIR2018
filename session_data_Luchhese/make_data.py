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

with open('dict/dict_char.json', 'r') as json_file:
    dict_char = json.load(json_file)
with open('dict/dict_word.json', 'r') as json_file:
    dict_word = json.load(json_file)
with open('dict/dict_click.json', 'r') as json_file:
    dict_click = json.load(json_file)


write_file = open("webis-smc-12.data", "w")
with open("webis-smc-12.txt") as ff:
    data = []
    for line_number, line in enumerate(ff):
        if line_number == 0:
            continue
        line = line.strip().split("\t")
        if len(line) < 2:
            continue
        data.append(line)

    for line_number, line in enumerate(data):
        user_id = line[0]
        query = line[1]
        time_str = line[2]
        rank = line[3]
        click = line[4]
        session_id = line[5]

        if rank == "":
            rank = "-1"

        word = []
        for i in query.split(" "):
            word.append(str(dict_word[i]))
        char = []
        for i in query:
            char.append(str(dict_char[i]))

        if line_number - 1 >= 0 and line_number - 1 < len(data):
            ts1 = int(time.mktime(time.strptime(time_str,'%Y-%m-%d %H:%M:%S')))
            ts2 = int(time.mktime(time.strptime(data[line_number - 1][2],'%Y-%m-%d %H:%M:%S')))
            diff1 = ts1 - ts2
            if user_id != data[line_number - 1][0]:
                diff1 = 36000
        else:
            diff1 = 36000

        if line_number + 1 >= 0 and line_number + 1 < len(data):
            ts1 = int(time.mktime(time.strptime(time_str,'%Y-%m-%d %H:%M:%S')))
            ts2 = int(time.mktime(time.strptime(data[line_number + 1][2],'%Y-%m-%d %H:%M:%S')))
            diff2 = ts2 - ts1
            if user_id != data[line_number + 1][0]:
                diff2 = 36000
        else:
            diff2 = 36000
            


        output = user_id + "\t" + session_id + "\t" + rank + "\t" + str(diff2) + "\t" + str(diff1) + "\t" + ','.join(word) + '\t' + ','.join(char)

        write_file.write(output + "\n")
