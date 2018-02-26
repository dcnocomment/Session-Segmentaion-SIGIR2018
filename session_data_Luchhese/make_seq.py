#!/usr/bin/python
#-*- coding: gb2312 -*-
import math
import sys 
import os
import time
import numpy as np
import json

def get_element(pool, i, default):
    if i >=0 and i < len(pool):
        return "\1".join(pool[i][2:7])
    else:
        return "\1".join(default)

SEQ_SIZE = 10
d = {}
padding = ["-1", "-1", "-1", "-1", "-1"]
feature = open("feature.dat", "w")
target = open("target.dat", "w")
with open("webis-smc-12.data") as ff:
    for line_number, line in enumerate(ff):
        line = line.strip().split("\t")
        user_id = line[0]
        if user_id not in d:
            d[user_id] = []
        d[user_id].append(line)


for user_id, content in d.items():
    for i in range(len(content) - 1):
        data = []
        for j in range(SEQ_SIZE):
            data.append(get_element(content, i - (SEQ_SIZE/2-1) + j, padding))
        tar = 0
        if content[i][1] == content[i+1][1]:
            tar = 1
        feature.write("\t".join(data) + "\n")
        target.write(str(tar) + "\n")
