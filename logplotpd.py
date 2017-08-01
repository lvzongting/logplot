#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

data = {}
item = ['entropy','output_rf','num_samples_rf','reward_predictor_err','nsamples_per_shard']
idx  = [3,5,7,9,11]
#item = ['entropy','output_rf','reward_predictor_err']
#idx  = [3,5,9]

for num,i in enumerate(item):
    data[i]=[];

#logfile = open('log.0')
logfile = open(sys.argv[1])
for line in logfile:
    line = re.split(', |:| \|\| |\n',line)
    if line[0] == 'shard':
        #print line[1]
        for num,i in enumerate(item):
            try:
                data[i].append(float(line[idx[num]]))
            except:
                continue

df = pd.DataFrame(data)
#print(df)
df.to_csv(sys.argv[1]+'.data', sep='\t')
