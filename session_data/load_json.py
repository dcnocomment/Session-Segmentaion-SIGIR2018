import json
import sys

name = sys.argv[1]

with open(name) as f:
    data_dict = json.load(f)

for k,v in data_dict.items():
    print str(v) + '\t' + str(k)
