
"""
__file__

    genFeat_cooccurrence_tfidf.py

__description__

    This file generates the following features for each run and fold, and for the entire training and testing set.

        1. tfidf for the following cooccurrence terms
            - query unigram/bigram & title unigram/bigram
            - query unigram/bigram & description unigram/bigram
            - query id & title unigram/bigram
            - query id & description unigram/bigram

        2. corresponding lsa (svd) version features

__author__

    xionglin <optimus1009@163.com>

"""

import re
import sys
import cPickle
import ngram
from feat_utils import dump_feat_name
from sklearn.decomposition import TruncatedSVD
from nlp_utils import stopwords, english_stemmer, stem_tokens, getTFV
sys.path.append("../")
from param_config import config

######################
## Pre-process data ##
######################
token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=config.cooccurrence_word_exclude_stopword,
                    encode_digit=False):
    token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    ## tokenize
    tokens = [x.lower() for x in token_pattern.findall(line)]
    ## stem
    tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]
    return tokens_stemmed


########################
## Cooccurrence terms ##
########################
def cooccurrence_terms(lst1, lst2, join_str):
    terms = [""] * len(lst1) * len(lst2)
    cnt =  0
    for item1 in lst1:
        for item2 in lst2:
            terms[cnt] = item1 + join_str + item2
            cnt += 1
	# 这里比如 terms = [A,B,C].则 " ".join(terms)的结果是 'A B C'
    res = " ".join(terms)
	#type(tes)  -> str
    return res


##################
## Extract feat ##
##################
def extract_feat(df):
    ## unigram
    print "generate unigram"
    df["query_unigram"] = list(df.apply(lambda x: preprocess_data(x["query"]), axis=1))
    df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
    df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
    ## bigram
    print "generate bigram"
    join_str = "_"
    df["query_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["query_unigram"], join_str), axis=1))
    df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
    df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
    ## 这里的 trigram 不需要，因为很多query不满足trigram的条件
    # join_str = "_"
    # df["query_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["query_unigram"], join_str), axis=1))
    # df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
    # df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

    ## cooccurrence terms
    join_str = "X"
    # query unigram
    df["query_unigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_unigram"], join_str), axis=1))
    df["query_unigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["title_bigram"], join_str), axis=1))
    df["query_unigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_unigram"], join_str), axis=1))
    df["query_unigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_unigram"], x["description_bigram"], join_str), axis=1))
    # query bigram
    df["query_bigram_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_unigram"], join_str), axis=1))
    df["query_bigram_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["title_bigram"], join_str), axis=1))
    df["query_bigram_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_unigram"], join_str), axis=1))
    df["query_bigram_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(x["query_bigram"], x["description_bigram"], join_str), axis=1))
    # query id
    df["query_id_title_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid"+str(x["qid"])], x["title_unigram"], join_str), axis=1))
    df["query_id_title_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid"+str(x["qid"])], x["title_bigram"], join_str), axis=1))
    df["query_id_description_unigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid"+str(x["qid"])], x["description_unigram"], join_str), axis=1))
    df["query_id_description_bigram"] = list(df.apply(lambda x: cooccurrence_terms(["qid"+str(x["qid"])], x["description_bigram"], join_str), axis=1))

        

if __name__ == "__main__":

    ############
    ## Config ##
    ############
    ## cooccurrence terms column names
	##  query unigram/bigram   <---------->    product_title unigram/bigram
	##  query unigram/bigram   <---------->	   product_description unigram/bigram
	##  query id               <---------->    product_title unigram/bigram
	##  query id               <---------->    product_description unigram/bigram
    column_names = [
        "query_unigram_title_unigram",
        "query_unigram_title_bigram",
        "query_unigram_description_unigram",
        "query_unigram_description_bigram",
        "query_bigram_title_unigram",
        "query_bigram_title_bigram",
        "query_bigram_description_unigram",
        "query_bigram_description_bigram",
        "query_id_title_unigram",
        "query_id_title_bigram",
        "query_id_description_unigram",
        "query_id_description_bigram",
    ]
    ## feature names  feat_names = [query_tfidf product_title_tfidf product_description_tfidf]
    feat_names = [ name+"_tfidf" for name in column_names ]
    ## file to save feat names
    feat_name_file = "%s/intersect_tfidf.feat_name" % config.feat_folder
	## cooccurrence_tfidf_ngram_range=(1,1)
    ngram_range = config.cooccurrence_tfidf_ngram_range

    svd_n_components = 100

    ###############
    ## Load Data ##
    ###############
    ## load data
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = cPickle.load(f)
    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)

    #######################
    ## Generate Features ##
    #######################
    print("==================================================")
    print("Generate co-occurrence tfidf features...")

    ## get cooccurrence terms
    extract_feat(dfTrain)
    extract_feat(dfTest)

    ######################
    ## Cross validation ##
    ######################
    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
                
            for feat_name,column_name in zip(feat_names, column_names):
                print "generate %s feat" % feat_name
                ## tfidf
                tfv = getTFV(ngram_range=ngram_range)
                X_tfidf_train = tfv.fit_transform(dfTrain.iloc[trainInd][column_name])
                X_tfidf_valid = tfv.transform(dfTrain.iloc[validInd][column_name])
                with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_tfidf_train, f, -1)
                with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_tfidf_valid, f, -1)

                ## svd
                svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
                X_svd_train = svd.fit_transform(X_tfidf_train)
                X_svd_test = svd.transform(X_tfidf_valid)
                with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                    cPickle.dump(X_svd_train, f, -1)
                with open("%s/valid.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
                    cPickle.dump(X_svd_test, f, -1)

    print("Done.")


    #################
    ## Re-training ##
    #################
    print("For training and testing...")
    path = "%s/All" % config.feat_folder
    for feat_name,column_name in zip(feat_names, column_names):
        print "generate %s feat" % feat_name
        tfv = getTFV(ngram_range=ngram_range)
        X_tfidf_train = tfv.fit_transform(dfTrain[column_name])
        X_tfidf_test = tfv.transform(dfTest[column_name])
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_tfidf_train, f, -1)
        with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_tfidf_test, f, -1)

        ## svd
        svd = TruncatedSVD(n_components=svd_n_components, n_iter=15)
        X_svd_train = svd.fit_transform(X_tfidf_train)
        X_svd_test = svd.transform(X_tfidf_test)
        with open("%s/train.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
            cPickle.dump(X_svd_train, f, -1)
        with open("%s/test.%s_individual_svd%d.feat.pkl" % (path, feat_name, svd_n_components), "wb") as f:
            cPickle.dump(X_svd_test, f, -1)

    print("Done.")

    ## save feat names
    print("Feature names are stored in %s" % feat_name_file)
    feat_names += [ "%s_individual_svd%d"%(f, svd_n_components) for f in feat_names ]
    dump_feat_name(feat_names, feat_name_file)

    print("All Done.")