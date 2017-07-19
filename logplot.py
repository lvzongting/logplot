import numpy as np
import re
import matplotlib.pyplot as plt
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

data = {}
item = ['entropy','output_rf','num_samples_rf','reward_predictor_err','nsamples_per_shard']
idx  = [3,5,7,9,11]
#item = ['entropy','output_rf','reward_predictor_err']
#idx  = [3,5,9]

for num,i in enumerate(item):
    data[i]=[];

logfile = open('log.0')
for line in logfile:
    line = re.split(', |:| \|\| |\n',line)
    if line[0] == 'shard':
        #print line[1]
        for num,i in enumerate(item):
            data[i].append(float(line[idx[num]]))

for num,i in enumerate(item):
    #print len(data[i]),type(data[i])
    #plt.figure(num)
    plt.subplot(len(idx),1,num+1)
    plt.plot(data[i],label=i,color=colors[num%len(colors)])
    plt.legend(loc="upper left")



plt.show()
