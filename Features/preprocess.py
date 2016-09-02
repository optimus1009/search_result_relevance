
"""
__file__

    preprocess.py

__description__

    This file preprocesses data.

__author__

    xionglin <optimus1009@163.com>
    
"""

import sys
import cPickle
import numpy as np
import pandas as pd
from nlp_utils import clean_text, pos_tag_text
sys.path.append("../")
from param_config import config

###############
## Load Data ##
###############
print("Load data...")

dfTrain = pd.read_csv(config.original_train_data_path).fillna("")
dfTest = pd.read_csv(config.original_test_data_path).fillna("")

#df.shape[0]返回的是数据的行数
num_train, num_test = dfTrain.shape[0], dfTest.shape[0]

print("Done.")


######################
## Pre-process Data ##
######################
print("Pre-process data...")


## test数据集中没有 median_relevance 和 relevance_variance这两列
dfTest["median_relevance"] = np.ones((num_test))
dfTest["relevance_variance"] = np.zeros((num_test))

## 插入样本索引值
dfTrain["index"] = np.arange(num_train)
dfTest["index"] = np.arange(num_test)

## one-hot encode the median_relevance
## n_classes = 4
for i in range(config.n_classes):   # i-> 0，1，2，3
    # median_relevance 的值为 1，2，3，4
    dfTrain["median_relevance_%d" % (i+1)] = 0
    dfTrain["median_relevance_%d" % (i+1)][dfTrain["median_relevance"]==(i+1)] = 1
    
## query ids
qid_dict = dict()
for i,q in enumerate(np.unique(dfTrain["query"]), start=1):
    qid_dict[q] = i
    
## insert query id
## map(func,seq1[,seq2]):将func作用于给定序列的每个元素，
## 并于一个列表来提供返回值
dfTrain["qid"] = map(lambda q: qid_dict[q], dfTrain["query"])
dfTest["qid"] = map(lambda q: qid_dict[q], dfTest["query"])

## clean text
#def clean_text(line, drop_html_flag=False):
#    names = ["query", "product_title", "product_description"]
#    for name in names:
#        l = line[name]
#        if drop_html_flag:
#            l = drop_html(l)
#        l = l.lower()
#        ## replace gb
#        for vol in [16, 32, 64, 128, 500]:
#            ## eg: 用16gb来替换 16 gb,
#            l = re.sub("%d gb"%vol, "%dgb"%vol, l)
#            l = re.sub("%d g"%vol, "%dgb"%vol, l)
#            l = re.sub("%dg "%vol, "%dgb "%vol, l)
#        ## replace tb
#        for vol in [2]:        
#            l = re.sub("%d tb"%vol, "%dtb"%vol, l)
#
#        ## replace other words
#        for k,v in replace_dict.items():
#            l = re.sub(k, v, l)
#        l = l.split(" ")
#
#        ## replace synonyms
#        l = replacer.replace(l)
#        l = " ".join(l)
#        line[name] = l
#    return line


## clean_text()主要用于对['query','product_title','product_descripition']
## 下面的数据进行清洗 
clean = lambda line: clean_text(line, drop_html_flag=config.drop_html_flag)
dfTrain = dfTrain.apply(clean, axis=1)
dfTest = dfTest.apply(clean, axis=1)

print("Done.")


###############
## Save Data ##
###############
print("Save data...")

## processed_train_data_path = "%s/train.processed.csv.pkl
## pickle.dump(obj, file[, protocol]) 
## If the protocol parameter is omitted, protocol 0 is used. 
## If protocol is specified as a negative value or HIGHEST_PROTOCOL, 
## the highest protocol version will be used.
with open(config.processed_train_data_path, "wb") as f:
    cPickle.dump(dfTrain, f, -1)
with open(config.processed_test_data_path, "wb") as f:
    cPickle.dump(dfTest, f, -1)
    
print("Done.")


"""
## pos tag text
dfTrain = dfTrain.apply(pos_tag_text, axis=1)
dfTest = dfTest.apply(pos_tag_text, axis=1)
with open(config.pos_tagged_train_data_path, "wb") as f:
    cPickle.dump(dfTrain, f, -1)
with open(config.pos_tagged_test_data_path, "wb") as f:
    cPickle.dump(dfTest, f, -1)
print("Done.")
"""
